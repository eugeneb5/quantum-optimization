import numpy as np

from functools import reduce

import matplotlib.pyplot as plt

from numpy.typing import ArrayLike



#take the necessary functions to reconstruct the problem H again

def sigma_z(n,j): #must be that n>
    
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


def H_ising(n,M,kappa):

    #M is the adjacency matrix
    h = np.zeros(n)

    for k in range(0,n): #range doesn't include n, goes from 0 to n-1 here!!

        total = 0

        for j in range(0,n):

            total += -(M[k,j]+M[j,k])

        h[k] = total + kappa

    
    #to make H_ising now
    H = np.zeros((2**n,2**n))
    H_2 = np.zeros((2**n,2**n))
    for k in range(1,n+1):   #1 to n


        H_2 += h[k-1]*sigma_z(n,k)
        

        for j in range(k+1,n+1): #k+1 to n

            H += (M[k-1,j-1]*sigma_z(n,k)@sigma_z(n,j)) 
    
    return (H+H_2)




#define variables:

M = np.array([[0,1,1,0,0],[0,0,1,0,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]])

k = 0.5

n = 5


#try the tine independent case to get the final wavefunc (expected to be |19>)

H = H_ising(n,M,k)

print(H.shape)

