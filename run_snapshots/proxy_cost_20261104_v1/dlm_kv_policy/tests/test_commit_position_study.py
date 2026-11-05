import unittest

import numpy as np

from dlm_kv_policy import commit_position_study as study

STOP = 9


def make_log(order, tokens, prompt_length=3, confidence=None):
    """Dense trajectory that commits position order[t] at step t, one per step.

    Confidence is fixed per position (higher = committed earlier), so the
    committed position always ranks first at every lag.
    """

    length = len(order)
    steps = length
    committed = np.zeros((steps, length), bool)
    for step, position in enumerate(order):
        committed[step, position] = True
    priority = np.zeros(length)
    for step, position in enumerate(order):
        priority[position] = -step
    conf = np.full((steps, length), np.nan, np.float32)
    top1 = np.full((steps, length), -1, np.int32)
    done = np.cumsum(committed, axis=0) - committed
    for step in range(steps):
        masked = done[step] == 0
        conf[step, masked] = priority[masked] if confidence is None else confidence[step][masked]
        top1[step, masked] = np.where(np.array(tokens)[masked] == STOP, STOP, 1)
    return study.TaskLog(
        dict(confidence=conf, top1=top1, committed=committed, dense_token_ids=np.array(tokens, np.int32),
             prompt_length=prompt_length, stop_ids=np.array([STOP])), 0)


class CommitPositionTest(unittest.TestCase):
    def setUp(self):
        # answer = positions 0-3, padding = 4-11; commits sweep left to right
        self.tokens = [5, 5, 5, 5] + [STOP] * 8
        self.log = make_log(list(range(12)), self.tokens)

    def test_masks_events_and_classes(self):
        log = self.log
        self.assertEqual(log.answer_end, 4)
        self.assertEqual(int(log.masked[0].sum()), 12)
        self.assertEqual(int(log.masked[5].sum()), 7)
        events = log.events()
        self.assertEqual(len(events), 11)  # step 0 cannot reuse anything
        self.assertEqual(log.event_class(3), 0)
        self.assertEqual(log.event_class(4), 1)

    def test_stale_rank_and_protected_sets(self):
        log = self.log
        for lag in (1, 4, 8):
            self.assertEqual(study.stale_rank(log, 6, 6, lag), 1)
        top = study.protected_set(log, 6, 2, "top", 1)
        self.assertEqual(np.flatnonzero(top).tolist(), [6])
        spread = study.protected_set(log, 6, 2, "top±1", 1)
        self.assertEqual(np.flatnonzero(spread).tolist(), [6, 7])  # position 5 is already committed
        recent = study.protected_set(log, 6, 2, "recent", 16)
        self.assertEqual(np.flatnonzero(recent).tolist(), [6])  # neighbors of steps 4-5 commits that are still masked
        split = study.protected_set(log, 2, 2, "split", 1)
        self.assertTrue(split[2])  # top answer-like position
        self.assertEqual(int(split[4:].sum()), study.PADDING_TOP)  # padding gets its own slots

    def test_locality(self):
        nearest, from_left = study.locality(self.log, 6, 6)
        self.assertEqual((nearest, from_left), (1, 0))
        late = make_log([0, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], self.tokens)
        nearest, from_left = study.locality(late, 1, 11)
        self.assertEqual((nearest, from_left), (11, 10))

    def test_coverage_and_selection(self):
        logs = [make_log(list(range(12)), self.tokens) for _ in range(4)]
        for i, log in enumerate(logs):
            log.index = i
        stats = study.analyze_logs(logs)
        everyone = np.ones(4, bool)
        for li in range(len(study.LAGS)):
            self.assertEqual(study.coverage(stats, everyone, li, study.VARIANTS.index("top"), 0, None), 1.0)
        chosen = study.select_rules(stats, np.arange(4))
        self.assertTrue(chosen[2]["selected"])
        self.assertEqual(chosen[2]["variant"], "recent")  # smallest set that still covers all commits here

    def test_missed_commit_lowers_coverage(self):
        conf = np.zeros((12, 12), np.float32)
        order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for step in range(12):
            conf[step] = np.arange(12)[::-1]  # stale confidence says left first...
        conf[:, 11] = 1000  # ...but position 11 always looks best, though it commits last
        log = make_log(order, self.tokens, confidence=conf)
        self.assertGreater(study.stale_rank(log, 3, 3, 2), 1)
        self.assertFalse(study.protected_set(log, 3, 2, "top", 1)[3])

    def test_rows_floor_and_truncation(self):
        log = self.log
        layer, head = study.step_rows(log, None, False)
        self.assertEqual(layer[0], 3 + 12)
        self.assertEqual(layer[5], 7 + 1)  # 7 masked + the row committed at step 4
        self.assertEqual(head[5], 7)
        tokens = [5, 5, STOP] + [STOP] * 9
        trunc_log = make_log(list(range(12)), tokens)
        layer_t, _ = study.step_rows(trunc_log, None, True)
        self.assertEqual(layer_t[2], 12 - 2 + 1)  # stop not yet committed at step 2's start
        self.assertEqual(layer_t[3], 1)  # only the stop token's own row (committed at step 2) is recomputed
        self.assertEqual(layer_t[4], 0)  # everything after the stop at 2 is frozen, including later commits
        reuse_rows, _ = study.step_rows(log, {"variant": "top", "m": 1, "window": 4}, False)
        self.assertAlmostEqual(reuse_rows[6], 1 + (6 - 1) / 4 + 1)  # 6 masked, 1 protected, 1 new


if __name__ == "__main__":
    unittest.main()
