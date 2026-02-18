from pymoo.core.sampling import Sampling
import numpy as np
import numpy.typing as npt


class BinaryRandomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs) -> npt.NDArray[np.bool_]:
        # Create a population of binary vectors (0s or 1s) of shape (n_samples, n_var)
        return np.random.randint(0, 2, size=(n_samples, problem.n_var)).astype(
            np.bool_
        )


class BinaryHybridSampling(Sampling):
    """
    Hybrid gen-0 sampler:
      - 1 exact default schedule (all-recompute baseline)
      - most candidates are sparse perturbations near baseline
      - remaining candidates are broad random samples
    """

    def __init__(
        self,
        default_vector: npt.NDArray[np.bool_] | None = None,
        near_baseline_fraction: float = 0.75,
        sparse_flip_fraction: float = 0.002,
        min_sparse_flips: int = 1,
        max_sparse_flips: int | None = None,
        include_default: bool = True,
    ) -> None:
        super().__init__()
        self.default_vector = default_vector
        self.near_baseline_fraction = near_baseline_fraction
        self.sparse_flip_fraction = sparse_flip_fraction
        self.min_sparse_flips = min_sparse_flips
        self.max_sparse_flips = max_sparse_flips
        self.include_default = include_default

    @staticmethod
    def _as_bool_vector(arr: npt.NDArray[np.bool_] | npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
        out = np.asarray(arr).astype(np.bool_).reshape(-1)
        return out

    def _resolve_default_vector(self, problem) -> npt.NDArray[np.bool_] | None:
        if self.default_vector is not None:
            vec = self._as_bool_vector(self.default_vector)
            if vec.size != problem.n_var:
                raise ValueError(
                    f"default_vector length {vec.size} does not match problem.n_var {problem.n_var}."
                )
            return vec

        if hasattr(problem, "default_schedule"):
            vec = self._as_bool_vector(problem.default_schedule)
            if vec.size != problem.n_var:
                raise ValueError(
                    f"problem.default_schedule length {vec.size} does not match problem.n_var {problem.n_var}."
                )
            return vec

        return None

    def _sample_sparse_neighbors(
        self,
        baseline: npt.NDArray[np.bool_],
        n_samples: int,
        n_var: int,
    ) -> npt.NDArray[np.bool_]:
        if n_samples <= 0:
            return np.empty((0, n_var), dtype=np.bool_)

        if self.max_sparse_flips is None:
            max_flips = int(np.ceil(self.sparse_flip_fraction * n_var))
        else:
            max_flips = self.max_sparse_flips
        max_flips = max(self.min_sparse_flips, max_flips)
        max_flips = min(n_var, max_flips)

        out = np.empty((n_samples, n_var), dtype=np.bool_)
        for i in range(n_samples):
            k = np.random.randint(self.min_sparse_flips, max_flips + 1)
            flip_idx = np.random.choice(n_var, size=k, replace=False)
            vec = baseline.copy()
            vec[flip_idx] = ~vec[flip_idx]
            out[i] = vec
        return out

    def _do(self, problem, n_samples, **kwargs) -> npt.NDArray[np.bool_]:
        n_var = int(problem.n_var)
        if n_samples <= 0:
            return np.empty((0, n_var), dtype=np.bool_)

        baseline = self._resolve_default_vector(problem)
        if baseline is None:
            # Fallback when no default schedule is available.
            return np.random.randint(0, 2, size=(n_samples, n_var)).astype(np.bool_)

        n_baseline = 1 if self.include_default and n_samples >= 1 else 0
        remaining = n_samples - n_baseline
        n_near = int(round(remaining * self.near_baseline_fraction))
        n_near = max(0, min(remaining, n_near))
        n_random = remaining - n_near

        parts: list[npt.NDArray[np.bool_]] = []
        if n_baseline > 0:
            parts.append(baseline.reshape(1, -1))
        if n_near > 0:
            parts.append(self._sample_sparse_neighbors(baseline, n_near, n_var))
        if n_random > 0:
            parts.append(
                np.random.randint(0, 2, size=(n_random, n_var)).astype(np.bool_)
            )

        X = np.vstack(parts).astype(np.bool_)
        if X.shape != (n_samples, n_var):
            raise ValueError(
                f"Hybrid sampler produced shape {X.shape}, expected {(n_samples, n_var)}."
            )
        return X
