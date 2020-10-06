#!/usr/bin/env python
# coding: utf-8

import pulp as p
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from IPython.display import display, clear_output
from tqdm import tqdm
import os

FIG_PATH = "./Image/M15"
M = 10

def create_problem(types):
    dim = len(types)
    supp = len(types[0])
    types = np.array(types).T.tolist()
    
    problem = p.LpProblem('Multi-Goods Monopolist', p.LpMaximize) 
    transfers = p.LpVariable.dicts("Transfers", range(supp),lowBound=0, cat='Continuous')
    allocations = p.LpVariable.dicts("Allocations", [(i,j) for i in range(supp) for j in range(dim)], lowBound=0, upBound=1, cat='Continuous')
    
    problem += 1/supp*p.lpSum(transfers)
    
    for i in range(supp):
        problem += M * allocations[(i,0)] + allocations[(i,1)] >= 0
        problem += -1/M * allocations[(i,0)] + allocations[(i,1)] >= 0
        problem += -M * allocations[(i,0)] + allocations[(i,1)] >= 1-M
        problem += 1/M * allocations[(i,0)] - allocations[(i,1)] >= 1/M - 1
    
    for i in range(supp):
        problem += p.lpSum([allocations[(i,j)] * types[i][j] for j in range(dim)]) - transfers[i] >= 0
        for k in range(supp):
            if i != k:
                problem += (p.lpSum([allocations[(i,j)] * types[i][j] for j in range(dim)]) - transfers[i] >=
                            p.lpSum([allocations[(k,j)] * types[i][j] for j in range(dim)]) - transfers[k])
   
    problem.solve()
    allocations_array = np.zeros((supp,dim))
    for i in range(supp):
        for j in range(dim):
            allocations_array[i,j] = allocations[(i,j)].value()
    allocations_array = allocations_array.T
    types_array = np.array(types).T
    if allocation_restriction(allocations_array.T):
        plt.scatter(types_array[0],types_array[1],color='red',s=8)
        plt.scatter(allocations_array[0],allocations_array[1],color='blue',s=4)
        plt.xlim(-.05, 1.05)
        plt.ylim(-.05, 1.05)
        plt.savefig(os.path.join(FIG_PATH,"{},{}.png").format(types,allocations_array))
        plt.clf()
        
def type_restriction(types):
    return not any([len(set(type)) <= 1 for type in types])

def allocation_restriction(allocations):
    temp_allocations = allocations
    temp_allocations = np.where(np.logical_or(temp_allocations == 0, temp_allocations == 1), False, True)
    return temp_allocations.all(axis=1).any(axis=0)

def allocations_iterator():
    grid = [x/5 for x in range(1,6)]
    return product(product(grid, repeat = 3), repeat = 2)

def main():
    for types in tqdm(allocations_iterator()):
        types = np.array(types).tolist()
        if type_restriction(types):
            allocations = create_problem(types)

# create_problem([[1/8, 0,1],[3/8,1/2,1]])

main()