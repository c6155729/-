

import numpy as np
from time import *
from matplotlib import pyplot as plt

bestV = 0
curW = 0
curV = 0
bestx = None


# 散点图
def show(w, v):
    plt.figure(figsize=(5, 5), dpi=100)
    plt.xlabel('wight')
    plt.ylabel('value')
    plt.title('散点图')
    plt.scatter(w, v)
    plt.show()


# 重量比排序
def wrather(w, v):
    m = []
    for i in range(0, len(w) - 1):
        n = w[i] / v[i]
        m.append(n)
    m = [round(i, 3) for i in m]
    m = sorted(m, reverse=True)
    print(m)


# 回溯法
def backtrack(i):
    begin_time = time()
    global bestV, curW, curV, x, bestx
    if i >= n:
        if bestV < curV:
            bestV = curV
            bestx = x[:]
    else:
        if curW + w[i] <= c:
            x[i] = True
            curW += w[i]
            curV += v[i]
            backtrack(i + 1)
            curW -= w[i]
            curV -= v[i]
        x[i] = False
        backtrack(i + 1)
    end_time = time()
    run_time = end_time - begin_time
    return '最大价值：' + str(bestV) + '\n耗时：' + str(run_time)


# 动态规划法
def bag(n, c, w, v):
    begin_time = time()
    res = [[-1 for j in range(c + 1)] for i in range(n + 1)]
    for j in range(c + 1):
        res[0][j] = 0
    for i in range(1, n + 1):
        for j in range(1, c + 1):
            res[i][j] = res[i - 1][j]
            if j >= w[i - 1] and res[i][j] < res[i - 1][j - w[i - 1]] + v[i - 1]:
                res[i][j] = res[i - 1][j - w[i - 1]] + v[i - 1]
    end_time = time()
    run_time = end_time - begin_time
    return res


# 贪心法
def tx(w, v, n, c):
    begin_time = time()
    data = np.array(w)
    idex = list(np.lexsort([data[:, 0], -1 * v[:, 1]]))
    status = [0] * n
    Tw = 0
    Tv = 0
    for i in range(n):
        if data[idex[i], 0] <= c:
            Tw += data[idex[i], 0]
            Tv += data[idex[i], 1]
            status[idex[i]] = 1
            c -= data[idex[i], 0]
        else:
            continue
    end_time = time()
    run_time = end_time - begin_time
    return Tv



if __name__ == '__main__':
    file = open('D:\软件工程\实验二\测试数据\\beibao2.in', 'r')
    string = file.read().strip()
    ss = string.split('\n')
    s = []
    file.close()
    w = []
    v = []
    w, v = np.loadtxt(ss, delimiter=' ', unpack=True)
    n = v[0]  # 物品的数量
    c = w[0]  # 物品的限制重量
    v = v[1:]
    w = w[1:]
    k = 0
    x = [False for i in range(int(n))]
    print(w)
    print(v)
    wrather(w, v)
    print("1.回溯法 \n2.贪心法 \n3.动态规划法 ")
    print("请输入你的选项：")
    input(k)
    if k==1:
        s = backtrack(0)
    if k==2:
        s = bag(n, c, w, v)
    if k==3:
        s = tx(w, v, n, c)
    file1 = open('D:\软件工程\实验二\测试数据\\return.txt', 'w')
    file1.write("".join(s))
    file1.close()
    show(w, v)
