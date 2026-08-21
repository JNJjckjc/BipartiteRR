import math
import pandas as pd

E=2
def Tinimoto_coefficient(x, y):
    return x * y / (x ** 2 + y ** 2 - x * y)


true_item = 3
other_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
similarities = {}
for i, item in enumerate(other_items, 1):
    similarity = Tinimoto_coefficient(true_item, item)
    similarities[item] = similarity


similarities_desc = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}

print("sim desc:", similarities_desc)


weights = {}
for i, (item, similarity) in enumerate(similarities_desc.items(), 1):
    weight_key = f's{i}'
    if i == 1:
        weight = math.exp(E)
    else:
        weight = 1
    weights[weight_key] = weight

print("W:", weights)


total_weight = sum(weights.values())


unit_weights = {f'w{i}': v / total_weight for i, v in enumerate(weights.values(), start=1)}


print("UNIT W:", unit_weights)

Q=0

# 计算效用 Q
for i, similarity in enumerate(similarities_desc.values()):
    weight_key = f'w{i+1}'
    unit_weight = unit_weights[weight_key]
    product = similarity * unit_weight
    Q += product

print(f"Q {Q:.4f}")


array = list(similarities_desc.values())

#∂Q/∂si
m = 1
for i in range(2, len(other_items) + 1):
    dQ_dsi = 0
    for j in range(1, len(other_items) + 1):
        if j != i:
            dQ_dsi += (array[i-1] - array[j-1]) * weights[f's{j}']
    dQ_dsi /= (total_weight ** 2)
    if dQ_dsi > 0:
        m = i
        weights[f's{i}'] = math.exp(E)
    else:
        break
    print(f"s{i}的∂Q/∂s{i} 为{dQ_dsi:.4f}")


total_weight = sum(weights.values())


print("UPDATE W:", weights)
print(f" m {m}")


unit_weights = {f'w{i}': v / total_weight for i, v in enumerate(weights.values(), start=1)}


print("UPDATE UNIT W:", unit_weights)

# UPDATE Q
Q=0
for i, similarity in enumerate(similarities_desc.values()):
    weight_key = f'w{i+1}'
    unit_weight = unit_weights[weight_key]
    product = similarity * unit_weight
    Q += product

print(f"UPDATE Q:{Q:.4f}")


