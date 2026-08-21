from scipy.optimize import brent, fmin, minimize
import numpy as np
from collections import defaultdict

def find_prefix(current_position, data):
    prefix = 44
    while True:
        count = 0
        for other in data:
            match_count = 0
            for i in range(prefix):
                if current_position[i] == other[i]:
                    match_count += 1
                else:
                    break

            if match_count >= prefix:
                count += 1
        if count >= 50:
            return prefix, count
        prefix -= 2

# 找到分组时最小的前缀长度（50 个位置左右）
def find_prefix2(current_position, data):
    prefix = 4
    while True:
        count = 0
        for other in data:
            match_count = 0
            for i in range(prefix):
                if current_position[i] == other[i]:
                    match_count += 1
                else:
                    break
            if match_count >= prefix:
                count += 1
        if count <= 50:
            return prefix, count
        prefix += 2

# 执行分组
def Optgroup(origin):
    data = []
    for item in origin:
        data.append(item[0])

    final_group = defaultdict(list)

    for loc1 in data:
        b0 = (0, 46)
        b1 = (0, 46)
        bnds = (b0, b1)

        con1 = {'type': 'ineq', 'fun': constraint1}
        cons = [con1]

        min_prefix, count = find_prefix2(loc1, data)
        print(min_prefix, count)

        x0 = np.array([min_prefix + 10, min_prefix])  # 2 个分组时，为每个组分配前缀长度
        res = minimize(objFun, x0, method='SLSQP', bounds=bnds, constraints=cons)
        # print(res.x)

        G_1 = []
        G_2 = []
        for loc2 in data:
            group = comparestring(loc1, loc2, res.x)
            if group == 1:
                G_1.append(loc2)
            if group == 2:
                G_2.append(loc2)
        final_group[loc1].append(G_1)
        final_group[loc1].append(G_2)

    return final_group


def objFun(x):
    dataor = ("data_gowalla_encode.txt")
    data = []
    for item in dataor:
        data.append(item[0])
    data = list(set(data))

    func = 0

    for loc1 in data:
        GLCP = {}
        for loc2 in data:
            group = comparestring(loc1, loc2, x)
            if group in GLCP:
                GLCP[group] += 1
            else:
                GLCP[group] = 1

        for i in range(2):
            if i + 1 in GLCP:
                func = func + x[i] * GLCP[i + 1]
            else:
                func = func
    return -func


def constraint1(x):
    return x[0] - x[1] + 1


def comparestring(loc1, loc2, beta):
    com = 0
    group = 0
    for i in range(len(loc1)):
        if loc1[i] == loc2[i]:
            com = com + 1
        else:
            break
    for i in range(2):
        if com >= beta[i]:
            group = i + 1
            break
    return group


def main():
    # 读取数据集文件
    with open('data_gowalla_encode.txt') as f:
        lines = f.readlines()

    # 数据预处理：将数据转换为位置编码
    data = [[line.strip().split(',')[0]] for line in lines]

    # 使用 Optgroup 函数进行位置分组
    groups = Optgroup(data)

    # 打印分组结果
    for loc, groups_in_loc in groups.items():
        print(f"位置 {loc} 分组为：")
        for i, group in enumerate(groups_in_loc, 1):
            print(f"  - 组 {i}: {group}")


if __name__ == '__main__':
    main()