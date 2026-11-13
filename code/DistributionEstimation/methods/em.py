"""Expectation-maximization for discrete distribution estimation."""

import numpy as np


def project_simplex(values):
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indexes = np.arange(1, values.size + 1)
    active = ordered - (cumulative - 1.0) / indexes > 0.0
    if not np.any(active):
        return np.full(values.size, 1.0 / values.size)
    rho = indexes[active][-1]
    threshold = (cumulative[active][-1] - 1.0) / rho
    return np.maximum(values - threshold, 0.0)


def em_step(estimate, observed, channel):
    report_probability = channel.T @ estimate
    ratio = observed / np.maximum(report_probability, 1e-300)
    return project_simplex(estimate * (channel @ ratio))


def estimate(observed, channel, max_iter=1000, tol=1e-8):
    """Estimate input probabilities from report frequencies and a channel.

    ``channel[i, j]`` is the probability of report ``j`` given input ``i``.
    """
    observed = np.asarray(observed, dtype=float)
    observed = observed / observed.sum()
    current = np.full(channel.shape[0], 1.0 / channel.shape[0])

    for _ in range(max_iter):
        updated = em_step(current, observed, channel)
        if np.linalg.norm(updated - current, ord=1) < tol:
            return updated
        current = updated
    return current
