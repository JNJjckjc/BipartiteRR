"""Mixture reduction for discrete distribution estimation."""

import math

import numpy as np

from .em import estimate as em_estimate


def _merge_groups(groups, start, width):
    merged = tuple(index for group in groups[start:start + width] for index in group)
    return groups[:start] + [merged] + groups[start + width:]


def _expand(weights, groups, size):
    result = np.zeros(size, dtype=float)
    for weight, group in zip(weights, groups):
        result[list(group)] = weight / len(group)
    return result / result.sum()


def _log_likelihood(observed, channel, weights, sample_size):
    fit = np.maximum(channel.T @ weights, 1e-300)
    return float(sample_size * np.dot(observed, np.log(fit)))


def _bic(log_likelihood, components, sample_size):
    return -2.0 * log_likelihood + max(0, components - 1) * math.log(sample_size)


def estimate(
    observed,
    channel,
    sample_size,
    min_components=None,
    merge_threshold=0.05,
    max_iter=1000,
    tol=1e-8,
):
    """Fit, merge low-mass adjacent components, and expand the estimate.

    ``channel`` contains one report distribution per ordered input component.
    Component groups are merged only with their immediate neighbors.
    """
    observed = np.asarray(observed, dtype=float)
    observed = observed / observed.sum()
    size = channel.shape[0]
    min_components = min_components or max(2, int(np.ceil(0.1 * size)))
    groups = [(index,) for index in range(size)]
    current_channel = np.asarray(channel, dtype=float).copy()
    weights = em_estimate(observed, current_channel, max_iter, tol)
    current_bic = _bic(_log_likelihood(observed, current_channel, weights, sample_size), len(groups), sample_size)

    while len(groups) > min_components:
        width = max(2, int(np.ceil(len(groups) / 2.0)))
        if width > len(groups) - min_components + 1:
            width = len(groups) - min_components + 1
        masses = np.convolve(weights, np.ones(width), mode="valid")
        start = int(np.argmin(masses))
        if masses[start] >= merge_threshold:
            break

        candidate_groups = _merge_groups(groups, start, width)
        candidate_channel = np.vstack([
            np.asarray(channel)[list(group)].mean(axis=0) for group in candidate_groups
        ])
        candidate = em_estimate(observed, candidate_channel, max_iter, tol)
        candidate_bic = _bic(
            _log_likelihood(observed, candidate_channel, candidate, sample_size),
            len(candidate_groups),
            sample_size,
        )
        if candidate_bic > current_bic:
            break
        groups, current_channel, weights, current_bic = (
            candidate_groups, candidate_channel, candidate, candidate_bic
        )

    return _expand(weights, groups, size)
