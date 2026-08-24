import os
import json
import numpy as np
import matplotlib.pyplot as plt
import math

DATA_DIR = r'./data_cache'


def _save_rank_data(tag, distribution, N, goal, x, results, ratio_or_eps, fixed_val):
    """保存排名数据以便后续快速重绘（JSON 格式，人可读）
    fixed_val: 当 tag='epsilon' 时为 ratio 值, tag='ratios' 时为 epsilon 值
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    fname = os.path.join(DATA_DIR, f'rank_ems_{tag}_{distribution}_N{N}_{goal}_{fixed_val}.json')
    save_dict = {
        'x': list(x),
        'N': N,
        'ratio_or_eps': ratio_or_eps,
        'distribution': distribution,
        'goal': goal,
        'tag': tag,
    }
    for atk in ['RIA', 'ROA']:
        for k, v in results[atk].items():
            save_dict[f'{atk}_{k}'] = list(v)
    with open(fname, 'w', encoding='utf-8') as fp:
        json.dump(save_dict, fp, indent=2, ensure_ascii=False)
    print(f'数据已缓存: {fname}')


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


# ==================== 排名折线图 ====================

def compute_rank(freq, target):
    """返回 target 在 freq 中的降序排名（1 = 最高，N = 最低）"""
    return int(np.where(np.argsort(-freq) == target - 1)[0][0] + 1)


def _plot_figure(x, x_label, results, attack, distribution, N, ratio_or_eps, tag, out_dir, goal):
    """通用绘图：单 Y 轴，pre 细空心 / post 实心 / ΔRank 虚线三角菱形实心"""
    mech_style = {
        'BRR': {'color': '#1f77b4', 'marker': 'o', 'label': 'BRR'},
        'GRR': {'color': '#d62728', 'marker': 's', 'label': 'GRR'},
    }
    delta_marker = {'BRR': '^', 'GRR': 'D'}

    fig, ax = plt.subplots(figsize=(8, 5))

    z_offsets = {'BRR': 3, 'GRR': 0}
    for mech, ms in mech_style.items():
        pre_key = f'{mech}-pre'
        post_key = f'{mech}-post'
        delta = np.abs(np.array(results[pre_key]) - np.array(results[post_key]))
        zo = z_offsets[mech]

        ax.plot(x, results[post_key],
                color=ms['color'], marker=ms['marker'], linestyle='-',
                markersize=10, markerfacecolor=ms['color'], markeredgewidth=0,
                linewidth=2, zorder=zo + 2,
                label=f'{ms["label"]}_post-attack')
        ax.plot(x, results[pre_key],
                color=ms['color'], marker=ms['marker'], linestyle='-',
                markersize=10, markerfacecolor='white', markeredgecolor=ms['color'],
                markeredgewidth=1.0, linewidth=2, zorder=zo + 3,
                label=f'{ms["label"]}_pre-attack')
        ax.plot(x, delta,
                color=ms['color'], marker=delta_marker[mech], linestyle='--',
                markersize=10, markerfacecolor=ms['color'], markeredgewidth=0,
                linewidth=2, zorder=zo + 4,
                label=f'{ms["label"]}_ΔRank')

    ax.set_xlabel(x_label, fontsize=22, fontweight='bold')
    ax.set_ylabel('Rank', fontsize=22, fontweight='bold')
    ax.set_title(f'EMS {attack} {goal}  ({distribution}, N={N}, {ratio_or_eps})',
                 fontsize=20)
    ax.tick_params(axis='both', labelsize=17)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    ax.legend(fontsize=11, loc='best', handlelength=4)

    fig.tight_layout()
    fname = os.path.join(out_dir, f'rank_ems_vs_{tag}_{distribution}_N{N}_{attack}_{goal}.png')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'已保存: {fname}')


def plot_rank_vs_epsilon(
    N=60,
    n_data=50000,
    ratio=0.1,
    epsilons=None,
    n_seeds=10,
    distribution='zipf',
    goal='Promotion',
    seed=0
):
    if epsilons is None:
        epsilons = [1, 2, 3, 4, 5, 6]

    np.random.seed(seed)
    data = generate_data(distribution, n=n_data, N=N)
    freq = np.bincount(data, minlength=N + 1)[1:]
    nonzero = np.where(freq > 0)[0]
    if goal == 'Promotion':
        target = int(nonzero[np.argmin(freq[nonzero])] + 1)
    else:
        target = int(np.argmax(freq) + 1)

    mechanisms = ['GRR', 'BRR']
    attacks = ['RIA', 'ROA']
    results = {atk: {} for atk in attacks}

    for mechanism in mechanisms:
        for attack in attacks:
            pre_key = f'{mechanism}-pre'
            post_key = f'{mechanism}-post'
            pre_vals, post_vals = [], []

            for eps in epsilons:
                perturb = grr_perturb if mechanism == 'GRR' else brr_perturb
                base_reports = perturb(data, N, eps)
                base_est = ems_recovery(base_reports, N, eps, mechanism)
                pre_vals.append(compute_rank(base_est, target))

                post_ranks = []
                for s in range(n_seeds):
                    np.random.seed(s)
                    attacked_reports = attack_reports(
                        base_reports, N, eps, mechanism,
                        attack, goal, target, ratio
                    )
                    attacked_est = ems_recovery(attacked_reports, N, eps, mechanism)
                    post_ranks.append(compute_rank(attacked_est, target))
                post_vals.append(np.mean(post_ranks))

            results[attack][pre_key] = pre_vals
            results[attack][post_key] = post_vals

    out_dir = f'./result_ems/{distribution}'
    _save_rank_data('epsilon', distribution, N, goal, epsilons, results,
                    f'ratio={ratio}', f'ratio{ratio}')
    for attack in attacks:
        _plot_figure(epsilons, 'ε', results[attack], attack, distribution, N,
                     f'ratio={ratio}', 'epsilon', out_dir, goal)


def plot_rank_vs_ratios(
    N=60,
    n_data=50000,
    epsilon=1.0,
    ratios=None,
    n_seeds=10,
    distribution='zipf',
    goal='Promotion',
    seed=0
):
    if ratios is None:
        ratios = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]

    np.random.seed(seed)
    data = generate_data(distribution, n=n_data, N=N)
    freq = np.bincount(data, minlength=N + 1)[1:]
    nonzero = np.where(freq > 0)[0]
    if goal == 'Promotion':
        target = int(nonzero[np.argmin(freq[nonzero])] + 1)
    else:
        target = int(np.argmax(freq) + 1)

    mechanisms = ['GRR', 'BRR']
    attacks = ['RIA', 'ROA']
    results = {atk: {} for atk in attacks}

    for mechanism in mechanisms:
        for attack in attacks:
            pre_key = f'{mechanism}-pre'
            post_key = f'{mechanism}-post'
            pre_vals, post_vals = [], []

            perturb = grr_perturb if mechanism == 'GRR' else brr_perturb
            base_reports = perturb(data, N, epsilon)
            base_est = ems_recovery(base_reports, N, epsilon, mechanism)
            pre_rank = compute_rank(base_est, target)

            for r in ratios:
                post_ranks = []
                for s in range(n_seeds):
                    np.random.seed(s)
                    attacked_reports = attack_reports(
                        base_reports, N, epsilon, mechanism,
                        attack, goal, target, r
                    )
                    attacked_est = ems_recovery(attacked_reports, N, epsilon, mechanism)
                    post_ranks.append(compute_rank(attacked_est, target))
                pre_vals.append(pre_rank)
                post_vals.append(np.mean(post_ranks))

            results[attack][pre_key] = pre_vals
            results[attack][post_key] = post_vals

    out_dir = f'./result_ems/{distribution}'
    _save_rank_data('ratios', distribution, N, goal, ratios, results,
                    f'ε={epsilon}', f'eps{epsilon}')
    for attack in attacks:
        _plot_figure(ratios, 'Fake User Ratio', results[attack], attack, distribution, N,
                     f'ε={epsilon}', 'ratios', out_dir, goal)

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
