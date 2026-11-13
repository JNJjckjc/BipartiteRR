"""Expectation-maximization with iterative smoothing."""

import numpy as np

from .em import em_step, project_simplex


def smooth(distribution):
    distribution = np.asarray(distribution, dtype=float)
    if distribution.size < 2:
        return project_simplex(distribution)

    result = np.empty_like(distribution)
    result[0] = (2.0 * distribution[0] + distribution[1]) / 3.0
    result[-1] = (distribution[-2] + 2.0 * distribution[-1]) / 3.0
    if distribution.size > 2:
        result[1:-1] = (
            0.25 * distribution[:-2]
            + 0.50 * distribution[1:-1]
            + 0.25 * distribution[2:]
        )
    return project_simplex(result)


def estimate(observed, channel, max_iter=1000, tol=1e-8):
    """Apply an EM update and a smoothing step until convergence."""
    observed = np.asarray(observed, dtype=float)
    observed = observed / observed.sum()
    current = np.full(channel.shape[0], 1.0 / channel.shape[0])

    for _ in range(max_iter):
        updated = smooth(em_step(current, observed, channel))
        if np.linalg.norm(updated - current, ord=1) < tol:
            return updated
        current = updated
    return current
