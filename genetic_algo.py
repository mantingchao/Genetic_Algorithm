#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:03:47 2020

@author: Manting
"""

import numpy as np

#%% linear regression
def F1(t):
    return 0.063 * (t ** 3) - 5.284 * (t ** 2) + 4.887 * t + 412 + np.random.normal(0, 1)
n = 1000
A = np.zeros((n, 5))
b = np.zeros((n, 1))
for i in range(n):
    t = np.random.random() * 100
    b[i] = F1(t)
    A[i, 0] = t ** 4
    A[i, 1] = t ** 3
    A[i, 2] = t ** 2
    A[i, 3] = t
    A[i, 4] = 1

# 解Ax = b，反推A0~A4，min(Ax - b)^2
x = np.linalg.lstsq(A, b)[0]
#print(x)


#%% non-linear
def F2(x, A, B, C, D):
    return A * (x ** B) + C * np.cos(D * x) + np.random.normal(0, 1, x.shape)

# 基因轉成參數
def gene2coef(gene):
    A = (np.sum(2 ** np.arange(10) * gene[0 : 10]) - 511) / 100
    B = (np.sum(2 ** np.arange(10) * gene[10 : 20]) - 511) / 100
    C = (np.sum(2 ** np.arange(10) * gene[20 : 30]) - 511)
    D = (np.sum(2 ** np.arange(10) * gene[30 : 40]) - 511) / 100
    return A, B, C, D

n = 1000

# genetic algorithm
T = np.random.random((n, 1)) * 100
b2 = F2(T, 0.6, 1.2, 100, 0.4)

pop = np.random.randint(0, 2, (10000, 40))
fit = np.zeros((10000, 1))
# 繁衍幾代
for generation in range(10):
    for i in range(10000):
        A, B, C, D = gene2coef(pop[i, : ])
        fit[i] = np.mean(abs(F2(T, A, B, C, D ) - b2)) # 取絕對值算平均，fit越小越好
    print(np.mean(fit))
    sortf = np.argsort(fit[ : , 0]) # 由小到大排留下index
    pop = pop[sortf, : ] # pop按照sortf排
    for i in range(100,10000): # 只拿前100個人去繁衍，取代後9900的人(適者生存，交配)
        fid = np.random.randint(0, 100)
        mid = np.random.randint(0, 100)
        while mid == fid: # 避免取到同一個值
            mid = np.random.randint(0, 100)
        # 隨機產生遮罩決定要copy誰的基，40個0或1，0 copy mom，1 copy father
        mask = np.random.randint(0,2 ,(1, 40)) 
        son = pop[mid, : ]
        father = pop[fid, : ]
        son[mask[0, : ] == 1] = father[mask[0, : ] == 1]
        pop[i, : ] = son # 取代第i個人
    # 產生1000個突變，0變1，1變0
    for i in range(1000):
        m = np.random.randint(0, 10000)
        n = np.random.randint(0, 40)
        pop[m, n] = 1 - pop[m, n]
    
for i in range(10000):
    A, B, C, D = gene2coef(pop[i, : ])
    fit[i] = np.mean(abs(F2(T, A, B, C, D ) - b2)) # 取絕對值算平均，fit越小越好
sortf = np.argsort(fit[ : , 0]) # 由小到大排留下index
pop = pop[sortf, : ] # pop按照sortf排

A, B, C, D = gene2coef(pop[0, : ])
print(A, B, C, D) # 求出結果與b2參數比較是否相近
    
    
    
    
    
    
