import numpy as np 
#import sympy as sp
from functools import reduce
from scipy.linalg import expm
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh   #for larger matrices!
import matplotlib.pyplot as plt
import networkx as nx
import os
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm






def sigma_z(n,j): #must be that n>j



    #assert n>j
     #j must be greater than two?

    #I_2 = np.array([1,0],[0,1])
    I_2 = np.eye(2)
    S_Z = np.array([[1,0],[0,-1]])

    if j == 1:  
        TP_1 = np.eye(1)
    else:
        TP_1 = reduce(np.kron, [I_2]*(j-1)) #this repeats the tensor product of I_2 j-1 times

    
    if j==n:
        TP_2 = np.eye(1)
    else:

        TP_2 = reduce(np.kron, [I_2]*(n-j))

    final = np.kron(np.kron(TP_1,S_Z),TP_2)
    
    return final


def sigma_x(n,j): #must be that n>j

    #assert n>j
     #j must be greater than two?

    #I_2 = np.array([1,0],[0,1])
    I_2 = np.eye(2)
    S_X = np.array([[0,1],[1,0]])

    if j == 1:  
        TP_1 = np.eye(1)
    else:
        TP_1 = reduce(np.kron, [I_2]*(j-1)) #this repeats the tensor product of I_2 j-1 times

    
    if j==n:
        TP_2 = np.eye(1)
    else:

        TP_2 = reduce(np.kron, [I_2]*(n-j))

    final = np.kron(np.kron(TP_1,S_X),TP_2)
    
    return final


def problem_hamiltonian(M, B, J, n):  #M is just a constant, B is a 1d array, J is a square array

    I = np.eye(2**n, dtype= complex)

    first_term = M*I

    second_term = np.zeros((2**n,2**n),dtype = complex)

    third_term = np.zeros((2**n,2**n),dtype = complex)

    for i in range(1,n+1):

        second_term += B[i-1]*sigma_z(n,i)
        for j in range(i+1,n+1):

            third_term += J[i-1,j-1]*(sigma_z(n,i)@sigma_z(n,j))

    return first_term - 0.5*second_term+0.5*third_term


def satisfiability_problem_generation(n,ratio, USA = True, satisfiability_ratio = False):

    J = np.zeros((n,n))

    B = np.zeros(n)

    criteria = False
    M=0

    if USA:
        
        while criteria == False:

            clause = np.sort(np.random.choice(np.arange(1, n+1), 3, replace=False))  #outputs an array e.g. [1,5,2]
            M+=1
            for index_1, element_1 in enumerate(clause):

                B[element_1]] += 1

                for index_2 in clause[index_1+1:3]:

                    J[index_1, index_2] +=1

            Hamiltonian = problem_hamiltonian(M,B,J,n)

            eigenvalues = np.linalg(Hamiltonian)

            if np.min(eigenvalues) == 0:  #check how many solutions there are and iterate through to check how many zeros there are

                print("not a unique satisfiability problem yet")

                check = 0

                for eigenvalue in eigenvalues:

                    if eigenvalue == 0:
                        check+= 1
                if check == 1:  #i.e. there is only one, unique solution
                    print("found USA. Number of clauses: "+str(M))
                    criteria = True
            else:
                print("has no unique solutions")

                criteria = True
            
        return problem_hamiltonian(M,B,J,n)
    
    
    if satisfiability_ratio:

        M = round(n*ratio)  #might have to mess around with this

        M = 









        






        

        



























