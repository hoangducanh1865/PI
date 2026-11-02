"""
Tests of the trajectory helpers behind EVID: actions computed from recorded poses integrate back into the same path,
and a perfect reading scores zero error.

Author: Duc Anh Hoang
"""

import numpy as np

from src.utils.trajectory import actions_from_states, integrate, trajectory_scores


def random_states(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    position = np.cumsum(rng.normal(0, 0.01, (16, 3)), axis=0) + [0.3, 0.0, 0.1]
    rotation = np.cumsum(rng.normal(0, 0.05, (16, 3)), axis=0)
    return np.concatenate([position, rotation], axis=1), rng.integers(0, 2, 16).astype(float)


def test_actions_integrate_back_to_the_path():
    state, gripper = random_states()
    path = integrate(state[0], actions_from_states(state, gripper))
    np.testing.assert_allclose(path, state[:, :3], atol=1e-9)


def test_perfect_reading_scores_zero_error():
    state, gripper = random_states(1)
    scores, _ = trajectory_scores(actions_from_states(state, gripper), state, gripper)
    assert scores["pos_err_cm"] < 1e-7 and scores["action_err"] < 1e-9
    assert abs(scores["ndtw_3d"] - 1) < 1e-6 and abs(scores["path_ratio_3d"] - 1) < 1e-9


def test_still_reading_scores_the_motion():
    state, gripper = random_states(2)
    scores, path = trajectory_scores(np.zeros((15, 7)), state, gripper)
    np.testing.assert_allclose(path, np.repeat(state[:1, :3], 16, axis=0))
    expected = 100 * np.linalg.norm(state[1:, :3] - state[0, :3], axis=1).mean()
    assert abs(scores["pos_err_cm"] - expected) < 1e-9
