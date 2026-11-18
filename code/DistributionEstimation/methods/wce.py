"""Window-constrained distribution estimation with adaptive regularization."""

import numpy as np

from .em import project_simplex


def build_matrix_a(n_categories, m):
    """Build the BRR recovery matrix A = W.T for non-periodic windows."""
    if n_categories < 2 or not 1 <= m <= n_categories:
        raise ValueError("n_categories must be >= 2 and m must be in [1, n_categories]")

    indices = np.arange(n_categories)
    forward = np.zeros((n_categories, n_categories), dtype=float)
    for center in range(n_categories):
        nearest = np.lexsort((indices, np.abs(indices - center)))[:m]
        forward[center, nearest] = 1.0
    return forward.T


def estimate_theta(observed, p, q, clip=False):
    """Estimate BRR window statistics; leave unclipped for unbiasedness."""
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observed must be a non-empty one-dimensional vector")
    if p <= q:
        raise ValueError("p must be greater than q")

    theta = (observed - q) / (p - q)
    return np.clip(theta, 0.0, 1.0) if clip else theta


def _build_difference_matrix(n_categories):
    difference = np.zeros((n_categories - 1, n_categories), dtype=float)
    rows = np.arange(n_categories - 1)
    difference[rows, rows] = -1.0
    difference[rows, rows + 1] = 1.0
    return difference


def _brr_probabilities(n_categories, m, epsilon):
    denominator = m * np.exp(epsilon) + n_categories - m
    q = 1.0 / denominator
    return float(np.exp(epsilon) * q), float(q)


def _objective(x, target, operator, difference, gamma):
    residual = operator @ x - target
    roughness = difference @ x
    return float(residual @ residual + gamma * (roughness @ roughness))


def _solve(target, operator, difference, gamma, max_iter, tol):
    hessian = operator.T @ operator + gamma * difference.T @ difference
    linear = operator.T @ target
    step = 1.0 / max(2.0 * np.linalg.norm(hessian, ord=2), 1e-12)
    x = np.full(operator.shape[1], 1.0 / operator.shape[1])
    y = x.copy()
    momentum = 1.0

    for _ in range(max_iter):
        updated = project_simplex(y - step * (hessian @ y - linear))
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        next_y = updated + ((momentum - 1.0) / next_momentum) * (updated - x)
        if np.linalg.norm(updated - x, ord=1) < tol:
            return updated
        x, y, momentum = updated, next_y, next_momentum
    return x


def _split_counts(counts, folds, rng):
    parts = np.zeros((folds, counts.size), dtype=np.int64)
    for index, count in enumerate(counts):
        parts[:, index] = rng.multinomial(int(count), np.full(folds, 1.0 / folds))
    return parts


def estimate(
    report_counts,
    n_categories,
    m,
    epsilon,
    gamma_candidates,
    folds=3,
    seed=0,
    max_iter=5000,
    tol=1e-6,
):
    """Select regularization by held-out report likelihood, then refit.

    Build the BRR recovery matrix and channel, select gamma by held-out report
    likelihood, then refit the estimate using all reports.
    """
    report_counts = np.asarray(report_counts, dtype=np.int64)
    if report_counts.ndim != 1 or report_counts.size != n_categories or report_counts.sum() <= 0:
        raise ValueError("report_counts must contain a positive total and match n_categories")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    operator = build_matrix_a(n_categories, m)
    difference = _build_difference_matrix(n_categories)
    p, q = _brr_probabilities(n_categories, m, epsilon)
    channel = q + (p - q) * operator.T

    candidates = np.asarray(gamma_candidates, dtype=float)
    if candidates.ndim != 1 or candidates.size == 0 or np.any(candidates < 0.0):
        raise ValueError("gamma_candidates must be a non-empty vector of non-negative values")

    partitions = _split_counts(report_counts, folds, np.random.default_rng(seed))
    losses = np.zeros(candidates.size, dtype=float)

    for fold in range(folds):
        validation = partitions[fold]
        training = partitions.sum(axis=0) - validation
        target = estimate_theta(training / training.sum(), p, q)
        validation_total = max(int(validation.sum()), 1)

        for index, gamma in enumerate(candidates):
            candidate = _solve(target, operator, difference, gamma, max_iter, tol)
            predicted_reports = np.maximum(channel.T @ candidate, 1e-300)
            losses[index] += -np.dot(validation, np.log(predicted_reports)) / validation_total

    selected_gamma = float(candidates[np.argmin(losses)])
    full_target = estimate_theta(report_counts / report_counts.sum(), p, q)
    return _solve(full_target, operator, difference, selected_gamma, max_iter, tol)
