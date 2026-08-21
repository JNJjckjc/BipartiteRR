import os
import random
import numpy as np
import PO
import test


def read_location_data(filename):
    with open(filename, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data


def calculate_pro(m, d, c, G):  # 计算扰动概率
    # 计算求和部分: sum((j-1) * |G(j)|) 从 j=2 到 j=m
    summation = 0
    for j in range(2, m+1):
            # print(len(G[j]))
            summation += (j - 1) * len(G[j-1])

    # 计算min
    denominator = (m - 1) * d * c - summation*(c-1)
    if denominator <= 0:
        raise ValueError("Denominator is zero, division by zero error.")

    min_val = (m - 1) / denominator
    # print(min_val)

    # 计算max
    max_val = min_val * c

    # 计算dif
    dif = (max_val - min_val) / (m - 1)

    # 生成pro列表
    pro = [max_val - i * dif for i in range(m)]
    # print(pro)

    # 统计空组的个数(empty)以及设置pro对应位置为0
    empty = 0
    for i in range(m):
        if len(G[i]) == 0:  # 如果G[i]为空
            pro[i] = 0
            empty += 1

    # 计算剩余非空组的概率和
    non_empty_sum = sum(pro)

    # 归一化剩余非空组的概率
    if non_empty_sum != 0:  # 防止除以零错误
        # 对于非零的 pro 值，归一化
        pro = [p / non_empty_sum if p != 0 else 0 for p in pro]

    # print(pro)    #查看概率情况

    # global procount
    # procount=procount+1
    # print(procount)

    return pro


# def calculate_pro(m, d, c, G):  # 计算扰动概率
#     # 计算求和部分: sum((j-1) * |G(j)|) 从 j=2 到 j=m
#     summation = 0
#     for j in range(2, m + 1):
#         if j <= len(G):
#             summation += (j - 1) * len(G[j - 1])  # G[j-1] 访问列表的第 j 项
#
#     # 计算min
#     denominator = (m - 1) * d * c - summation
#     if denominator == 0:
#         raise ValueError("Denominator is zero, division by zero error.")
#
#     min_val = (m - 1) / denominator
#
#     # 计算max
#     max_val = min_val * c
#
#     # 计算dif
#     dif = (max_val - min_val) / (m - 1)
#
#     # 生成pro列表
#     pro = [max_val - i * dif for i in range(m)]
#
#     # 统计空组的个数(empty)以及设置pro对应位置为0
#     empty = 0
#     for i in range(m):
#         if len(G[i]) == 0:  # 如果G[i]为空
#             pro[i] = 0
#             empty += 1
#
#     # 计算剩余非空组的概率和
#     non_empty_sum = sum(pro)
#
#     # 归一化剩余非空组的概率
#     if non_empty_sum != 0:  # 防止除以零错误
#         # 对于非零的 pro 值，归一化
#         pro = [p / non_empty_sum if p != 0 else 0 for p in pro]
#     # print(pro)
#
#     # global procount
#     # procount=procount+1
#     # print(procount)
#
#     return pro

#获得每个位置对应的概率，返回一个“位置：[概率]”的字典
def calculate_pro_dict(m, d, c, G_list):
    result_dict = {}
    for key, value in G_list.items():
        # print(value)
        pro = calculate_pro(m, d, c, value)
        result_dict[key] = pro

    # for G in G_list:
    #     pro = calculate_pro(m, d, c, G)
    #     result_dict[G] = pro
    return result_dict


# SRR扰动函数
def SRR(user, GLCP, probabilities, m):
    r = random.uniform(0, 1)  # 生成一个0到1之间的随机数
    s = 0
    j = 0
    i = 0
    count=0
    psum=0.0
    p=0.0

    # 选择哪个组
    while j < m:
        s += probabilities[j]

        if r <= s:  # 如果随机数落在当前组的概率区间内
            if not GLCP[j]:
                j=j+1
                continue

            sample = random.choice(GLCP[j])  # 从该组中随机选一个位置
            return sample
        else:
            j += 1  # 如果没有选中当前组，继续检查下一个组

    # 如果没有返回，默认返回第一个组的第一个位置
    return GLCP[0][0]


def computer_Qloss(encodetxt,weizhitxt,GLCP,pro_dict):
    data = test.files_to_dict(encodetxt, weizhitxt) #data是编码对应的位置，用于计算distance
    domain = read_location_data(encodetxt)  #domain是全部编码位置
    sum=0.0
    for local in domain:   #对全部位置遍历
        result = test.get_values_by_key(GLCP, local)  #获得可能发布的位置
        # probabilities = calculate_pro(m, len(data), c, GLCP[local])
        probabilities=pro_dict.get(local)
        for local_fabu in result:   #对可能发布位置遍历
            xh = test.get_float_list_from_dict(data, local_fabu)  #获取发布位置，用于计算d
            zi = test.get_float_list_from_dict(data, local)  #获取位置，用于计算d
            distance = test._distance(zi, xh)   #计算distance
            g_index,g_size=test.find_group(GLCP, local, local_fabu)  #找到发布位置对应的组下标，和组中位置个数，用于计算f
            # print(probabilities[g_index])

            fpro=probabilities[g_index]/g_size  #计算f（位置属于的组概率除以组大小）
            res=fpro*distance
            sum=sum+res   #计算Qloss
    # print(len(domain))
    Qloss=sum/len(domain)
    # print("!!!!!!!!!")
    # print(Qloss)
    return Qloss

def computer_Pr(encodetxt,weizhitxt,GLCP,probabilities):
    data = test.files_to_dict(encodetxt, weizhitxt)
    domain = read_location_data(encodetxt)
    sum = 0.0
    Pr_dict= {}
    for local in domain:
        groups_fenzi = test.get_values_by_key(GLCP, local)
        for local_fabu in groups_fenzi:
            g_index, g_size = test.find_group(GLCP, local, local_fabu)
            fpro_fenzi = len(domain) * probabilities[g_index] / g_size

            for key, list in GLCP.items():
                for sublist in list:
                    if local_fabu in sublist and local_fabu not in Pr_dict:
                        index, size = test.find_group(GLCP, key, local_fabu)
                        sum = sum + len(domain) * probabilities[index] / size


            Pr = fpro_fenzi / sum
            Pr_dict.setdefault(local_fabu, Pr)
    return Pr_dict



def calculate_f(x_fabu, x, GLCP,probabilities):
    result = test.get_values_by_key(GLCP,x)
    if x_fabu not in result:
        return 0
    else:
        # probabilities = calculate_pro(m, len(data), c, GLCP[x])
        g_index, g_size = test.find_group(GLCP, x, x_fabu)
        return probabilities[g_index] / g_size
    # probabilities = calculate_pro(m, len(data), c, GLCP[x])
    # if test.find_group(GLCP, x, x_fabu)==-1:
    #     return 0
    # else:
    #     g_index, g_size = test.find_group(GLCP, x, x_fabu)
    #     return probabilities[g_index] / g_size

def calculate_posterior_probs(local_fabu, local, domain,GLCP,pro_dict):
    probabilities = pro_dict[local]
    denominator = sum([calculate_f(local_fabu, x, GLCP, probabilities) for x in domain])
    if denominator != 0:
        fpro = calculate_f(local_fabu, local, GLCP, probabilities)
        return fpro/denominator
    else:
        return 0

def get_posterior_dict(domain ,GLCP,pro_dict):
    posterior_dict = {}
    i=0
    for local_fabu in domain:
        for local in domain:
            posterior=calculate_posterior_probs(local_fabu, local, domain,GLCP,pro_dict)
            posterior_dict[(local_fabu, local)]=posterior
        print(i)
        i=i+1
    return posterior_dict



# def calculate_posterior_probs(local_fabu, local, domain,GLCP,pro_dict):
#     # posterior_probs = []
#     probabilities = pro_dict[local]
#     denominator = sum([calculate_f(local_fabu, x, GLCP, probabilities) for x in domain])
#     if denominator != 0:
#         fpro = calculate_f(local_fabu, local, GLCP, probabilities)
#         return fpro/denominator
#     else:
#         return 0


# def calculate_posterior_probs(local_fabu, local, GLCP,pro_dict):
#     # posterior_probs = []
#     result_keys=test.get_keys_by_value(local_fabu,GLCP)
#     probabilities = pro_dict[local]
#     denominator = sum([calculate_f(local_fabu, x, GLCP, probabilities) for x in result_keys])
#     if denominator != 0:
#         fpro = calculate_f(local_fabu, local, GLCP, probabilities)
#         return fpro/denominator
#     else:
#         return 0

    # for local in domain:
    #     # probabilities=calculate_pro(m, len(domain), c, GLCP[local])
    #     probabilities=pro_dict[local]
    #     denominator = sum([calculate_f(local_fabu, x,GLCP,probabilities) for x in domain])
    #     if denominator!= 0:
    #         fpro = calculate_f(local_fabu, local, GLCP,probabilities)
    #         posterior_probs.append(fpro / denominator)
    #     else:
    #         posterior_probs.append(0)

    # return posterior_probs






def find_argmin(data,domain, local_fabu, distance_func,posterior_dict):

    # data = test.files_to_dict(encodetxt, weizhitxt)  # data是编码对应的位置，用于计算distance
    error=0.0
    min_error = float('inf')
    estimated_x = None

    for y in domain:
        for local  in domain:
            posterior=posterior_dict[(local_fabu, local)]
            error=error+posterior*distance_func(test.get_float_list_from_dict(data, y), test.get_float_list_from_dict(data,local))
        if error<min_error:
            min_error=error
            estimated_x=y
        error=0.0

    return estimated_x

def find_argmin_dict(encodetxt, weizhitxt,domain, distance_func,posterior_dict):
    data = test.files_to_dict(encodetxt, weizhitxt)
    extimated_x_dict={}
    count=1

    for local_fabu in domain:
        extimated_x_dict[local_fabu]=find_argmin(data,domain, local_fabu,distance_func,posterior_dict)
        print(count," ","发布位置",local_fabu,"的最小预期推断误差位置为",extimated_x_dict[local_fabu])
        count=count+1


    file_path = os.path.join("extimated_1,5", "[min_prefix+10,min_prefix+6,min_prefix].txt")
    with open(file_path, "w") as file:
        file.write(str(extimated_x_dict))

    return extimated_x_dict




# def computer_ExpErr(encodetxt, weizhitxt, GLCP,pro_dict):
#
#     data = test.files_to_dict(encodetxt, weizhitxt)
#     domain = read_location_data(encodetxt)
#     print("domain:",len(domain))
#
#     result_dict={}
#     sum=0.0
#     min_sum=0.0
#     sum_exp_err = 0.0
#     for local_fabu in domain:
#         # posterior_probs = calculate_posterior_probs(local_fabu, domain, GLCP,pro_dict)
#         # estimated_x = find_argmin(data,domain, posterior_probs, test._distance)
#         min_sum_dist = float('inf')
#         for local in domain:
#             # result = test.get_values_by_key(GLCP, local)
#             # probabilities = calculate_pro(m, len(data), c, GLCP[local])
#             # posterior=calculate_posterior_probs(local_fabu, local, GLCP,pro_dict)
#             estimated_x=find_argmin(data,domain, local_fabu, GLCP,pro_dict,test._distance)
#             probabilities = pro_dict.get(local)
#             if test.find_group(GLCP, local, local_fabu)==-1:
#                 fpro=0
#             else:
#                 g_index, g_size = test.find_group(GLCP, local, local_fabu)
#                 fpro = probabilities[g_index] / g_size
#             sum =sum + fpro * test._distance(test.get_float_list_from_dict(data, estimated_x),test.get_float_list_from_dict(data,local))
#
#         result_dict[estimated_x] = sum/len(domain)
#         min_sum=min(result_dict.values())
#         sum_exp_err=sum_exp_err + min_sum
#             # 计算当前组合的预期推断误差
#             # sum_dist = sum([fpro * test._distance(test.get_float_list_from_dict(data, estimated_x), test.get_float_list_from_dict(data,x)) for x in domain])
#             # min_sum_dist = min(min_sum_dist, sum)
#
#         # sum_exp_err += min_sum_dist
#
#     ExpErr = sum_exp_err / len(domain)
#     return ExpErr



def computer_ExpErr(encodetxt, weizhitxt, GLCP,pro_dict,estimated_x_txt):

    data = test.files_to_dict(encodetxt, weizhitxt)
    domain = read_location_data(encodetxt)

    with open(estimated_x_txt, "r") as file:
        content = file.read()
        estimated_x_dict = eval(content)

    sum=0.0
    for local_fabu in domain:
        estimated_x = estimated_x_dict[local_fabu]
        for local in domain:
            probabilities = pro_dict.get(local)
            if test.find_group(GLCP, local, local_fabu)==-1:
                fpro=0
            else:
                g_index, g_size = test.find_group(GLCP, local, local_fabu)
                fpro = probabilities[g_index] / g_size
            sum =sum + fpro * test._distance(test.get_float_list_from_dict(data, estimated_x),test.get_float_list_from_dict(data,local))

    ExpErr = sum / len(domain)
    return ExpErr

# 执行SRR扰动
def main(input_file, output_file, epsilon, m, c):
    # 假设的GLCP
    # GLCP = {
    #     '0111100100000110010001010100101001110000110011': [['0111100100000110010001010100101001110000110011'], ['0000000100000100111011111111000110101001010100','0111100100000100111011111100100011111010001111'], ['0111100100000100111011111100100011111010000010']],
    #     '0111100100000100111011111100100011111010000010': [['0111100100000100111011111100100011111010000010'], ['0000000100000100111011111111000110101001010100','0111100100000100111011111100100011111010001111'], ['0111100100000110010001010100101001110000110011']],
    #     '0000000100000100111011111111000110101001010100': [['0000000100000100111011111111000110101001010100'], ['0000000100000100111011111111000110101001010100'], ['0000000100000100111011111111000110101001010100','0111100100000100111011111100100011111010001111']],
    #     '0111100100000100111011111100100011111010001111': [['0111100100000100111011111100100011111010001111'], ['0000000100000100111011111111000110101001010100'], ['0000000100000100111011111111000110101001010100','0111100100000100111011111100100011111010001111']]
    # }


    # 读取数据集文件,得到完成分组后的GLCP
    with open('data_gowalla_encode.txt') as f:
        lines = f.readlines()
    # 数据预处理：将数据转换为位置编码
    data = [[line.strip().split(',')[0]] for line in lines]  # 假设每个位置信息只有一行，且以逗号分隔
    # 使用 Optgroup 函数进行位置分组
    GLCP = dict(PO_2.Optgroup(data))
    # print("GGGGGGGGGGGGGGG:")
    # print(GLCP)
    # 读取位置数据
    data = read_location_data(input_file)

    # 计算每个组的扰动概率
    # probabilities = compute_probabilities(epsilon, m, c)
    # print("PPPPPPPPPPPPPPP")
    # print(probabilities)

    # 存储扰动后的位置
    perturbed_locations = []

    pro_dict=calculate_pro_dict(m, len(data), c, GLCP)


    # 对每个位置执行SRR扰动
    for location in data:
        if location in GLCP:
            # pro=calculate_pro(m,len(data),c,GLCP[location])
            perturbed_location = SRR(location, GLCP[location], pro_dict.get(location), m)
            perturbed_locations.append(perturbed_location)  # 存储扰动后的结果

    # 将扰动后的结果写入输出文件
    with open(output_file, 'w') as f:
        for location in perturbed_locations:
            f.write(location + '\n')


    Qloss=computer_Qloss("dataset_gowalla_encode.txt","data_gowalla.txt",GLCP,pro_dict)
    print("Qloss:")




    # posterior_dict=get_posterior_dict(data, GLCP, pro_dict)
    #
    # argmin_list={}
    # argmin_list=find_argmin_dict("data_gowalla_encode.txt", "data_gowalla.txt",data, test._distance,posterior_dict)

    # ExpErr = computer_ExpErr("data_gowalla_encode.txt", "data_gowalla.txt", GLCP,pro_dict,"extimated_xs/[min_prefix+8,min_prefix+6,min_prefix].txt")
    # print("ExpErr:")
    # print(ExpErr)



    # Pr_dict = computer_Pr("data_gowalla_encode.txt","data_gowalla.txt",GLCP,probabilities)
    # # print(Pr_dict)

















# 运行主函数
if __name__ == '__main__':
    input_file = 'data_gowalla_encode.txt'  # 输入文件路径
    output_file = 'SRR_result.txt'  # 输出文件路径
    epsilon = 5 # 隐私预算
    m = 2  # 组数
    c = np.e ** epsilon  # 概率比率

    procount=0

    # 计算组数m的公式
    d = 1000
    #d = 3000
    print("m:",(2*(np.e**epsilon*d-np.e**(epsilon+1)))/((np.e**epsilon-1)*d))


    main(input_file, output_file, epsilon, m, c)
