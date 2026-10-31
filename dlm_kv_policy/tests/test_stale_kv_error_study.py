from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.stale_kv_error_study import (
    COMBOS,
    FOLDS,
    METHODS,
    POLICIES,
    StaleRowReplay,
    add_held_out,
    add_rows,
    assign_folds,
    empty_held_out,
    empty_sums,
    fit,
    fit_methods,
    linear_maps,
    load_fit,
    path_check,
    report,
    residual_energy,
    row_types,
)
from test_backend_cpu import FakeModel
from test_replay_compounding import greedy_sample_tokens

# The sampler looks up sample_tokens in the module that defines the model class.
sample_tokens = greedy_sample_tokens


class RotatingFakeModel(FakeModel):
    """FakeModel with a real (position-dependent) rotation, so pre- and post-RoPE keys differ."""

    def __init__(self) -> None:
        super().__init__()

        def rotary_emb(values, position_ids):
            angles = position_ids[..., None].float() * torch.tensor([0.3, 0.7])
            return torch.cat([angles.cos()] * 2, -1), torch.cat([angles.sin()] * 2, -1)

        self.model.rotary_emb = rotary_emb


def fake_replay(model, policies):
    dense = DiffuCoderActiveTokenKVBackend(model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
    replay = StaleRowReplay(
        model, dense, policies,
        SimpleNamespace(steps=10, gen_length=10, temperature=0.0, top_p=1.0, top_k=None,
                        algorithm="entropy", algorithm_temperature=0.0),
        None,
    )
    return replay


class StaleRowReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = RotatingFakeModel().eval()
        for parameter in self.model.parameters():
            parameter.data.mul_(3)
        self.prompt = torch.tensor([[4, 5, 6]])
        self.row_type = torch.tensor([0, 0, 0] + [1] * 6 + [2] * 4)

    def test_full_refresh_has_no_stale_rows_and_refresh_4_does(self) -> None:
        policies = {
            "every_step": DKVDecodeRefreshPolicy(cache_reloading_step=1),
            "refresh_4": DKVDecodeRefreshPolicy(cache_reloading_step=4),
        }
        replay = fake_replay(self.model, policies)
        sums = {name: empty_sums(2, 1, 4, "cpu") for name in policies}
        replay.run(self.prompt, torch.ones_like(self.prompt), self.row_type, lambda name, *rows: add_rows(sums[name], *rows))
        self.assertEqual(sums["every_step"]["rows"].sum().item(), 0)
        self.assertGreater(sums["refresh_4"]["rows"].sum().item(), 0)
        self.assertGreater(sums["refresh_4"]["ee"].diagonal(dim1=-2, dim2=-1).sum().item(), 0)

    def test_compares_keys_before_rope(self) -> None:
        replay = fake_replay(self.model, {"refresh_4": DKVDecodeRefreshPolicy(cache_reloading_step=4)})
        layer = self.model.model.layers[0]
        seen = []

        def sink(name, layer_id, kind, c, e, combo):
            if layer_id == 0 and kind == 0:
                seen.append(c)

        replay.run(self.prompt, torch.ones_like(self.prompt), self.row_type, sink)
        # A stale prompt row's layer-0 key is k_proj of its embedding, unrotated.
        expected = layer.self_attn.k_proj(self.model.model.embed_tokens(self.prompt[0])).double()
        torch.testing.assert_close(seen[0][0, :3], expected)
        rotated = replay.backends["refresh_4"].cache.keys[0][0, 0, 1:3].double()
        self.assertFalse(torch.allclose(rotated, expected[1:3]))


class FitTests(unittest.TestCase):
    def sums_for(self, c, e, combo):
        sums = empty_sums(1, 1, c.shape[-1], "cpu")
        add_rows(sums, 0, 0, c[None], e[None], combo)
        return sums

    def test_full_affine_fit_recovers_a_known_map(self) -> None:
        torch.manual_seed(0)
        c = torch.randn(2000, 6, dtype=torch.float64) + 3
        matrix, intercept = torch.randn(6, 6, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
        e = c @ matrix.T + intercept
        fitted = fit_methods(self.sums_for(c, e, torch.zeros(2000, dtype=torch.long)))
        torch.testing.assert_close(fitted["m4_matrix"][0, 0, 0, 0], matrix, rtol=0.05, atol=0.05)
        torch.testing.assert_close(fitted["m2_shift"][0, 0, 0, 0], e.mean(0))

    def test_residual_energy_from_sums_matches_direct(self) -> None:
        torch.manual_seed(1)
        c, e = torch.randn(50, 4, dtype=torch.float64), torch.randn(50, 4, dtype=torch.float64)
        combo = torch.randint(0, 3, (50,))
        sums = self.sums_for(c, e, combo)
        maps = linear_maps(fit_methods(sums))
        for weights, intercept in maps.values():
            from_sums = residual_energy(sums, weights, intercept)[0, 0, 0]
            for cell in range(3):
                rows = combo == cell
                w, b = weights[0, 0, 0, cell], intercept[0, 0, 0, cell]
                prediction = c[rows] * w + b if w.dim() == 1 else c[rows] @ w.T + b
                self.assertAlmostEqual(from_sums[cell].item(), (e[rows] - prediction).square().sum().item(), places=8)

    def test_held_out_sums_match_direct_and_fit_beats_no_correction(self) -> None:
        torch.manual_seed(2)
        c = torch.randn(1, 300, 4, dtype=torch.float64)
        e = 0.5 * c + 0.1 * torch.randn_like(c)
        combo = torch.randint(0, COMBOS, (300,))
        fitted = fit_methods(self.sums_for(c[0], e[0], combo))
        held = empty_held_out(1, 1, "cpu")
        add_held_out(held, fitted, 0, 0, c, e, combo)
        in_sample = torch.stack([residual_energy(self.sums_for(c[0], e[0], combo), *pair) for pair in linear_maps(fitted).values()])
        torch.testing.assert_close(held["residual"][:, 0, 0], in_sample[:, 0, 0])
        self.assertEqual(held["residual"].shape[0], len(METHODS))
        self.assertLess(held["residual"][-1].sum().item(), 0.1 * held["energy"].sum().item())


def run_synthetic_pipeline(output_dir: Path, errors) -> dict:
    """fit and report on synthetic stale rows: errors(c) gives e for rows c [1, rows, head_dim]."""

    torch.manual_seed(0)
    layers, head_dim, tasks, rows_per_task, combo = 2, 128, 8, 300, 3  # combo 3: age 2-3, prompt
    folds = [index % FOLDS for index in range(tasks)]
    data = {}
    for index in range(tasks):
        for layer in range(layers):
            for kind in range(2):
                c = torch.randn(1, rows_per_task, head_dim, dtype=torch.float64) + 1
                data[index, layer, kind] = (c, errors(c), torch.full((rows_per_task,), combo))
    for name in POLICIES:
        for fold in range(FOLDS):
            sums = empty_sums(layers, 1, head_dim, "cpu")
            for (index, layer, kind), rows in data.items():
                if folds[index] == fold:
                    add_rows(sums, layer, kind, *rows)
            (output_dir / "sums" / name).mkdir(parents=True, exist_ok=True)
            torch.save(sums, output_dir / "sums" / name / f"fold{fold}.pt")
    with redirect_stdout(io.StringIO()):
        fit(output_dir)
        for name in POLICIES:
            (output_dir / "eval" / name).mkdir(parents=True, exist_ok=True)
            for index in range(tasks):
                fitted = load_fit(output_dir / "fit" / name / f"fold{folds[index]}.npz", "cpu")
                held = empty_held_out(layers, 1, "cpu")
                for layer in range(layers):
                    for kind in range(2):
                        add_held_out(held, fitted, layer, kind, *data[index, layer, kind])
                arrays = {key: value.numpy() for key, value in held.items()}
                np.savez(output_dir / "eval" / name / f"task_{index:03d}.npz", task_id=f"T/{index}", fold=folds[index], **arrays)
        report(output_dir, tasks)
    return json.loads((output_dir / "report.json").read_text())


class SyntheticPipelineTests(unittest.TestCase):
    def test_planted_map_is_recovered_held_out(self) -> None:
        matrix = torch.randn(128, 128, dtype=torch.float64, generator=torch.Generator().manual_seed(5)) / 128**0.5
        with tempfile.TemporaryDirectory() as directory:
            result = run_synthetic_pipeline(Path(directory), lambda c: c @ matrix.T + 0.1 * torch.randn_like(c))
        scores = result["policies"]["refresh_32"]["prompt_and_answer"]
        self.assertGreater(scores["methods"]["M4"]["fer_mean"], 0.9)
        self.assertEqual(scores["first_moment"], "systematic")
        # The pipeline checks need GPU data, so without it the outcome is not trusted.
        self.assertTrue(result["outcome"].startswith("UNTRUSTED"))

    def test_pure_noise_is_not_predictable_held_out_although_it_fits_in_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_synthetic_pipeline(Path(directory), lambda c: torch.randn_like(c))
        scores = result["policies"]["refresh_32"]["prompt_and_answer"]
        m4 = scores["methods"]["M4"]
        self.assertGreater(m4["in_sample_fer"], 0.02)
        self.assertLess(m4["fer_mean"], 0.0)
        self.assertLess(m4["ci_low"], 0.0)
        self.assertEqual(scores["first_moment"], "not systematic")
        self.assertIn("C: neither", result["outcome"])


class PathCheckTests(unittest.TestCase):
    def first_step(self, difference_energy):
        layers = 3
        return {
            "rows": np.full((layers, 2, 1), 10.0),
            "difference_energy": np.full((layers, 2, 1), difference_energy),
            "dense_energy": np.full((layers, 2, 1), 1000.0),
            "difference_sum": np.zeros((layers, 2, 1, 4)),
        }

    def test_a_path_difference_that_is_large_against_stale_error_fails(self) -> None:
        stale_energy, stale_rows = np.full((3, 2), 100.0), np.full((3, 2), 10.0)  # 10 per row
        self.assertTrue(path_check(self.first_step(0.01), stale_energy, stale_rows)["passed"])
        failed = path_check(self.first_step(1.0), stale_energy, stale_rows)  # 0.1 per row: 1% of stale
        self.assertAlmostEqual(failed["mean_share_of_stale_error"], 0.01)
        self.assertFalse(failed["passed"])  # the layer-0 difference, 1e-3 of dense, is too large


class SetupTests(unittest.TestCase):
    def test_folds_are_balanced_by_dataset(self) -> None:
        ids = [f"HumanEval/{i}" for i in range(32)] + [f"Mbpp/{i}" for i in range(32)]
        folds = assign_folds(ids)
        for fold in range(FOLDS):
            self.assertEqual(sum(f == fold for f, i in zip(folds, ids) if i.startswith("HumanEval")), 8)
            self.assertEqual(sum(f == fold for f, i in zip(folds, ids) if i.startswith("Mbpp")), 8)
        self.assertEqual(folds, assign_folds(ids))

    def test_row_types_split_answer_from_padding_at_the_first_stop(self) -> None:
        self.assertEqual(row_types(2, [7, 8, 0, 9, 0], {0}, "cpu").tolist(), [0, 0, 1, 1, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
