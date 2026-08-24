import os
import json
import numpy as np
import matplotlib.pyplot as plt
import math



def generate_data(distribution='zipf', n=5000, N=30, zipf_s=1.3):
    if distribution == 'uniform':
        return np.random.randint(1, N + 1, size=n)
    elif distribution == 'normal':
        mu, sigma = N / 2, N / 6
        return np.clip(np.random.normal(mu, sigma, n).round(), 1, N).astype(int)
    elif distribution == 'exponential':
        scale = N / 5
        return np.clip(np.random.exponential(scale, n).round(), 1, N).astype(int)
    elif distribution == 'zipf':
        ranks = np.arange(1, N + 1)
        probs = 1 / ranks ** zipf_s
        probs /= probs.sum()
        return np.random.choice(ranks, size=n, p=probs)
    else:
        raise ValueError


def find_optimal_m(N, epsilon):
    e_eps = math.exp(epsilon)
    m1_num = math.sqrt(N ** 2 * e_eps + 0.25 * (1 - e_eps) ** 2) - (N - e_eps / 2 + 0.5)
    m1_den = e_eps - 1
    m1 = int(m1_num / m1_den) if m1_den != 0 else 1

    if N % 2 == 0:
        i = int(N / (math.exp(epsilon / 2) + 1) + 1)
    else:
        num = math.sqrt(e_eps * (N ** 2 - 1) + 1) - N
        den = e_eps - 1
        i = int(num / den + 1) if den != 0 else 1

    m2 = i + 1 if i % 2 == 0 else i
    return max(1, min(min(m1, m2), N))


def grr_perturb(data, N, epsilon):
    e_eps = np.exp(epsilon)
    p = e_eps / (e_eps + N - 1)
    q = 1 / (e_eps + N - 1)

    perturbed = []
    for x in data:
        probs = np.full(N, q)
        probs[x - 1] = p
        perturbed.append(np.random.choice(np.arange(1, N + 1), p=probs))
    perturbed_arr = np.array(perturbed)
    counts = np.bincount(perturbed_arr, minlength=N + 1)[1:N + 1]

    return counts


def grr_probability_matrix(N, epsilon):
    e_eps = np.exp(epsilon)
    p = e_eps / (e_eps + N - 1)
    q = 1 / (e_eps + N - 1)
    P = np.full((N, N), q)
    np.fill_diagonal(P, p)
    return P


def brr_perturb(data, N, epsilon):
    m = find_optimal_m(N, epsilon)
    e_eps = np.exp(epsilon)
    p_high = e_eps / (m * e_eps + N - m)
    p_low = 1 / (m * e_eps + N - m)

    perturbed = []
    for x in data:
        distances = np.abs(np.arange(1, N + 1) - x)
        high = np.argsort(distances)[:m]
        probs = np.full(N, p_low)
        probs[high] = p_high
        perturbed.append(np.random.choice(np.arange(1, N + 1), p=probs))
    perturbed_arr = np.array(perturbed)
    counts = np.bincount(perturbed_arr, minlength=N + 1)[1:N + 1]

    return counts


def brr_probability_matrix(N, epsilon):
    m = find_optimal_m(N, epsilon)
    e_eps = np.exp(epsilon)
    p_high = e_eps / (m * e_eps + N - m)
    p_low = 1 / (m * e_eps + N - m)

    P = np.full((N, N), p_low)
    for i in range(N):
        distances = np.abs(np.arange(1, N + 1) - (i + 1))
        high = np.argsort(distances)[:m]
        P[i, high] = p_high
    return P


def ems_recovery(report_freq, N, epsilon, mechanism, max_iter=1000):
    """
    EMS 频率恢复：EM 反卷积 + S-step 邻域平滑，不使用伪计数。
    S-step: 每个点取 [0.25, 0.5, 0.25] 的邻域加权平均，抑制长尾尖峰。
    """
    total = report_freq.sum()
    x_est = np.ones(N) / N
    P = (
        grr_probability_matrix(N, epsilon)
        if mechanism == 'GRR'
        else brr_probability_matrix(N, epsilon)
    )

    for _ in range(max_iter):
        weighted = np.zeros(N)
        for y in range(N):
            num = P[:, y] * x_est
            posterior = num / num.sum()
            weighted += report_freq[y] * posterior

        x_new = weighted / total

        # S-step: 邻域平滑
        x_smoothed = np.copy(x_new)
        x_smoothed[0] = 0.5 * x_new[0] + 0.5 * x_new[1]
        x_smoothed[-1] = 0.5 * x_new[-1] + 0.5 * x_new[-2]
        for i in range(1, N - 1):
            x_smoothed[i] = 0.25 * x_new[i - 1] + 0.5 * x_new[i] + 0.25 * x_new[i + 1]
        x_smoothed /= x_smoothed.sum()

        x_est = x_smoothed

    return x_est


def attack_reports(perturb_freq, N, epsilon, mechanism,
                   attack, goal, target, ratio=0.05):
    """
    attack: 'RIA' 或 'ROA'
    goal: 'Promotion' 或 'Demotion'
    target: 目标值 (1~N)
    ratio: 假用户比例
    """
    n_real = perturb_freq.sum()
    n_fake = int(n_real * ratio)

    perturb = grr_perturb if mechanism == 'GRR' else brr_perturb

    real_reports = perturb_freq
    fake_reports = np.zeros(N, dtype=int)

    competitors = np.arange(1, N + 1)
    competitors = competitors[competitors != target]

    if attack == 'RIA':
        if goal == 'Promotion':
            fake_reports = perturb(np.full(n_fake, target, dtype=int), N, epsilon)
        else:
            fake_value = np.random.choice(competitors, size=n_fake, replace=True)
            fake_reports = perturb(fake_value, N, epsilon)
    else:  # ROA
        if goal == 'Promotion':
            fake_reports = np.zeros(N, dtype=int)
            fake_reports[target - 1] = n_fake
        else:
            fake_reports = np.bincount(
                np.random.choice(competitors, size=n_fake, replace=True),
                minlength=N + 1
            )[1:]

    return real_reports + fake_reports


def frequency_attack_gain(before, after, target):
    return after[target - 1] - before[target - 1]


def ranking_attack_gain(before, after, target):
    rb = np.where(np.argsort(-before) == target - 1)[0][0] + 1
    ra = np.where(np.argsort(-after) == target - 1)[0][0] + 1
    return rb - ra


if __name__ == "__main__":
    N = 60
    n_data = 50000
    ratio = 0.1
    epsilons = [1, 2, 3, 4, 5, 6]
    n_seeds = 10
    seed = 0
    epsilon = 1.0
    ratios = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]

    distributions = ['uniform', 'normal', 'exponential', 'zipf']
    for goal in ['Promotion']:
        for dist in distributions:
            print(f'\n===== {dist} | {goal} =====')
            plot_rank_vs_epsilon(N, n_data, ratio, epsilons, n_seeds, dist, goal, seed)
            plot_rank_vs_ratios(N, n_data, epsilon, ratios, n_seeds, dist, goal, seed)
    print('\n全部完成.')
