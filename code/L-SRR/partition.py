from sqlalchemy.sql.functions import random
import random
import matplotlib.pyplot as plt

def files_to_dict(file_a, file_b):
    with open(file_a, 'r') as a, open(file_b, 'r') as b:
        dictionary = {line_a.strip(): line_b.strip() for line_a, line_b in zip(a, b)}
    return dictionary

def draw_1(Sets):
    colors = [
                 'red',
                 'blue',
                 'green',
                 'yellow',
                 'black',
                 'white',
                 'purple',
                 'orange',
                 'cyan',
                 'magenta',
                 'lime',
                 'pink',
                 'brown',
                 'navy',
                 'maroon',
                 'olive',
                 'teal',
                 'lavender',
                 'gold',
                 'silver',
                 'coral',
                 'salmon',
                 'tan',
                 'khaki',
                 'plum',
                 'indigo',
                 'beige',
                 'turquoise',
                 'violet',
                 'azure',
                 'peachpuff',
                 'chocolate',
                 'lightgray',
             ] * ( len(Sets)// 7 + 1)
    i=0
    max_x = 0
    count = 0
    for dataset in Sets:
        x = []
        y = []
        for data in dataset:
            x.append(data[0])
            y.append(data[1])
            if max_x<data[0]:
                max_x = data[0]
        c=colors[i]
        i+=1
        plt.scatter(x, y,color = c,s=40,alpha=0.5,cmap="viridis",edgecolors='white')
        plt.xticks(())
        plt.yticks(())
        count +=1
    for set in Sets:
        max_y = 0
        max_x = 0
        min_y = 1000
        min_x = 1000
        for data in set:
            x = data[0]
            y = data[1]
            if max_x<x:
                max_x = x
            if max_y <y:
                max_y=y
            if min_y >y:
                min_y = y
            if min_x>x:
                min_x= x
    plt.show()

def find_prefix(current_position, data):
    group=[]
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
                    break
            if match_count >= prefix:
                count += 1
                group.append(other)
        if count >= 6 :              #!!!! 改变这里>=的数字，越大每个分区的位置数越多 （6，8 较好）
            return prefix, count , group
        else:
            group.clear()
        prefix -= 2

def find_suffix(prefix, group, data):
    for item in group:
        subprefix, subcount, subgroup=find_prefix(item, data)
        if subprefix < prefix:
            print(subprefix,subcount,subgroup)
            group=group + subgroup
    return group

def fusion(dictSets,group,prefix):
    keys=[]
    for key, value in dictSets.items():
        if prefix<len(key) and group[0][:prefix]==key[:prefix]:
            keys.append(key)

    for key in keys:
        del dictSets[key]
        print("合并")


def Optgroup(origin):
    data = []
    dictSets={}
    for item in origin:
        data.append(item[0])

    d=len(data)
    print(d)
    remainder=data

    while len(remainder)>0:
        prefix, count, group = find_prefix(random.choice(remainder), data)
        fusion(dictSets,group,prefix)

        group=find_suffix(prefix, group, data)
        print("long:",len(group))

        remainder=list(set(remainder)-set(group))

        dictSets[group[0][:prefix]]=group

    return dictSets

def main():
    with open('data_gowalla_encode.txt') as f:
    #with open('dataset_TSMC2014_NYC_encode.txt') as f:
        lines = f.readlines()

    data = [[line.strip().split(',')[0]] for line in lines]

    count=0
    num=0
    dictSets = Optgroup(data)

    for key, value in dictSets.items():
        print('第',count+1,'组的位置个数：',len(value))
        num=num+len(value)
        count=count+1
    print('共计',count,'组')
    print('共计',num,'个位置')

    Sets=[]
    for key, value in dictSets.items():
        Sets.append(value)

    #print(Sets)     #!!!! Sets是最终分区结果（二进制编码）

    Sets2 = Sets

    domain = files_to_dict("data_gowalla_encode.txt", "data_gowalla.txt")
    for set in Sets2:
        for i in range(len(set)):
            set[i] = domain[set[i]]

    result = []
    for sub_list in Sets2:
        new_sub_list = []
        for item in sub_list:
            parts = [float(part.strip()) for part in item.split(',')]
            new_sub_list.append(parts)
        result.append(new_sub_list)


        #print(result)
        #print("")#!!!! result是最终分区结果（经纬度）

    draw_1(result)



if __name__ == '__main__':
    main()