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


def unique_satisfiability_problem_generation(n,ratio=0, USA = True, satisfiability_ratio = False, save_mode = True, no_sat = False):

    J = np.zeros((n,n))

    B = np.zeros(n)

    criteria = False
    M=0
    attempts = 0
    if USA:
        
        while criteria == False:

            clause = np.sort(np.random.choice(np.arange(0, n), 3, replace=False))  #outputs an array e.g. [1,5,2]
            M+=1
            for index_1, element_1 in enumerate(clause):

                B[element_1] += 1

                for element_2 in clause[index_1+1:3]:

                    J[element_1, element_2] +=1

            Hamiltonian = problem_hamiltonian(M,B,J,n)

            eigenvalues = eigh(Hamiltonian)[0]

            if np.min(eigenvalues) == 0:  #check how many solutions there are and iterate through to check how many zeros there are

                print("not a unique satisfiability problem yet. Number of clauses: "+str(M))

                check = 0

                for eigenvalue in eigenvalues:

                    if eigenvalue == 0:
                        check+= 1

                        
                if check == 1:  #i.e. there is only one, unique solution
                    print("found USA. Number of clauses: "+str(M))
                    criteria = True
                else:
                    print("number of solutions: " +str(check))
            elif no_sat:
                print("has no unique solutions")

                criteria = True
            else:
                M-=1
                print("no USA, will remove clause and repeat. Clause count: "+str(M))
                print("lowest eigenvalue: "+str(np.min(eigenvalues)))
            
                attempts +=1

                if attempts >=5:
                    print("terminating program, failed to find a USA")
                    return

                for index_1, element_1 in enumerate(clause):

                    B[element_1] -= 1

                    for element_2 in clause[index_1+1:3]:

                        J[element_1, element_2] -=1


            
        return problem_hamiltonian(M,B,J,n)
    
    
    elif satisfiability_ratio:

        M = round(n*ratio)  #might have to mess around with this

        for i in range(M):
            clause = np.sort(np.random.choice(np.arange(0, n), 3, replace=False))

            for index_1, element_1 in enumerate(clause):

                B[element_1] += 1

                for element_2 in clause[index_1+1:3]:

                    J[element_1, element_2] +=1

        return problem_hamiltonian(M,B,J,n)
    

def Time_dependent_Hamiltonian(n,t,t_max,H_p):

    A = lambda t, t_max: 1 - (t/t_max)

    B = lambda t, t_max: t/t_max

    val = np.zeros((2**n,2**n), dtype = complex)
    for i in range(1,n+1):
        val -= sigma_x(n,i)

    H_t = A(t,t_max)*val + B(t,t_max)*H_p

    return H_t


def Hamiltonian_spectrum(n,t_max,q,H_p,number_of_eigenvalues = 2):

    if number_of_eigenvalues < 2:
        print("terminating program, number_of_eigenvalues entry incorrect")
        return
    
    dt = t_max/(q+1)

    eigenvalues_set = np.zeros((q+1, number_of_eigenvalues))

    eigenvalue_range = np.linspace(0,number_of_eigenvalues-1, number_of_eigenvalues, dtype=int)

    for i in range(0,q+1):

        h = Time_dependent_Hamiltonian(n,dt*i,t_max,H_p)

        eigenvalues = np.sort(eigh(h)[0])

        for index in eigenvalue_range:

            eigenvalues_set[i,index] = eigenvalues[index]

    x_val = np.linspace(0,1,q+1)
    eigenvalue_difference = eigenvalues_set[:,1]-eigenvalues_set[:,0]  #this assumes number of eigen_values is at least 2
    print(np.min(eigenvalue_difference))

    plt.title("Example problem Hamiltonian eigenvalue Spectrum")
    for index in eigenvalue_range:
        plt.plot(x_val, eigenvalues_set[:,index], label = "eigenvalue "+str(index+1))
    plt.xlabel("Time parameter , s")
    plt.ylabel("Energy eigenvalue")
    plt.xticks([0,0.5,1])
    plt.legend()
    plt.show()

        
n = 6   #minimum of 4!
t_max = 50
q = 150

USA = unique_satisfiability_problem_generation(n, ratio = 0.8, USA = False, satisfiability_ratio= True)
#USA = unique_satisfiability_problem_generation(n)

Hamiltonian_spectrum(n, t_max, q, USA, number_of_eigenvalues = 4)






        






        

        



























