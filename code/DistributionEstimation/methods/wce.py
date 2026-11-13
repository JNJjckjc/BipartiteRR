"""Window-constrained distribution estimation with adaptive regularization."""

import numpy as np

from .em import project_simplex


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
    channel,
    statistic_matrix,
    operator,
    difference,
    gamma_candidates,
    folds=3,
    seed=0,
    max_iter=5000,
    tol=1e-6,
):
    """Select regularization by held-out report likelihood, then refit.

    ``statistic_matrix`` maps report frequencies to aggregate statistics;
    ``operator`` maps an input distribution to the same statistics.
    """
    report_counts = np.asarray(report_counts, dtype=np.int64)
    candidates = np.asarray(gamma_candidates, dtype=float)
    partitions = _split_counts(report_counts, folds, np.random.default_rng(seed))
    losses = np.zeros(candidates.size, dtype=float)

    for fold in range(folds):
        validation = partitions[fold]
        training = partitions.sum(axis=0) - validation
        target = statistic_matrix @ (training / training.sum())
        validation_total = max(int(validation.sum()), 1)

        for index, gamma in enumerate(candidates):
            candidate = _solve(target, operator, difference, gamma, max_iter, tol)
            predicted_reports = np.maximum(channel.T @ candidate, 1e-300)
            losses[index] += -np.dot(validation, np.log(predicted_reports)) / validation_total

    selected_gamma = float(candidates[np.argmin(losses)])
    full_target = statistic_matrix @ (report_counts / report_counts.sum())
    return _solve(full_target, operator, difference, selected_gamma, max_iter, tol)
