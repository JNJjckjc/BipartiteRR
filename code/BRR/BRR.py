import math

def Tinimoto_coefficient(x, y):
    return x * y / (x ** 2 + y ** 2 - x * y)

other_items = list(range(1,21))

m_list=[]
m=1
for true_item in range(1, 21):
    print("true_item:",true_item)
    similarities = {}
    for item in other_items:
        similarity = Tinimoto_coefficient(true_item, item)
        similarities[item] = similarity

    similarities_desc = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}
    print("排序后的相似度字典:", similarities_desc)
    weights = {}
    for i, (item, similarity) in enumerate(similarities_desc.items(), 1):
        weight_key = f's{i}'
        if i == 1:
            weight = math.exp(1)
        else:
            weight = 1
        weights[weight_key] = weight
    # 输出权值
    print("权值:", weights)
    total_weight = sum(weights.values())
    unit_weights = {f'w{i}': v / total_weight for i, v in enumerate(weights.values(), start=1)}
    # 输出单位化权值
    print("单位化权值:", unit_weights)
    Q = 0
    for i, similarity in enumerate(similarities_desc.values()):
        weight_key = f'w{i+1}'
        unit_weight = unit_weights[weight_key]
        product = similarity * unit_weight
        Q += product
    print(f"效用 Q 的值为 {Q:.4f}")
    array = list(similarities_desc.values())
    m = 1
    for i in range(2, len(other_items) + 1):
        dQ_dsi = 0
        for j in range(1, len(other_items) + 1):
            if j != i:
                dQ_dsi += (array[i-1] - array[j-1]) * weights[f's{j}']
        dQ_dsi /= (total_weight ** 2)
        if dQ_dsi > 0:
            m = i
            weights[f's{i}'] = math.exp(1)
        else:
            break
        print(f"对s{i}的偏导数 ∂Q/∂s{i} 为{dQ_dsi:.4f}")
    total_weight = sum(weights.values())
    m_list.append(m)
    # 输出更新后的权值,m

m_min = min(m_list)
print(f"\n m : {m_list}")
print(f"m_min = {m_min} \n")

final_weights_s = {}
for i in range(1, len(other_items) + 1):
    if i <= m_min:
        final_weights_s[f's{i}'] = math.exp(1)
    else:
        final_weights_s[f's{i}'] = 1


total_final_weight = sum(final_weights_s.values())
final_weights_w = {f'w{i}': value / total_final_weight
                   for i, value in enumerate(final_weights_s.values(), start=1)}


print(" s_i:")
for key, value in final_weights_s.items():
    print(f"  {key}: {value:.4f}")

print("\n w_i:")
for key, value in final_weights_w.items():
    print(f"  {key}: {value:.6f}")
