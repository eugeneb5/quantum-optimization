import numpy as np 
#import sympy as sp
from functools import reduce
from scipy.linalg import expm
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh   #for larger matrices!
import matplotlib.pyplot as plt
import networkx as nx







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

def a_coefficients_false(t,t_max,s_x):

    
    s = t/t_max

    if s < s_x:
        return 1
    else:
        1 - ((s-s_x)/(1-s_x))

def a_coefficients_true(t,t_max,s_x):    #ALTER LATER! to test how it works!!

    #test what c_x =/= 0 would be like?

    s = t/t_max

    c_x = 0

    c_1 = 0

    if s<s_x:
        return c_x
    
    else:
        return c_x +(c_1-c_x)*((s-s_x)/(1-s_x))

def h_z(J, n, kappa = 0.5, R=1):  #need to tinker with this one! we want to normalise to R?

    h = np.zeros(n)

    for k in range(0,n): 

        total = 0

        for j in range(0,n):

            total += -(J[k,j]+J[j,k])

        h[k] = total + kappa


    #to normalize:

    h_norm = (h*R) / np.max(np.abs(h))

    return h_norm

def b_coefficients_true(t,t_max,s_x):
    
    s = t/t_max
    return (s-s_x)/(1-s_x)

def b_coefficients_false(t,t_max,s_x):

    s = t/t_max
    if s<s_x:
        return 0
    else:
        return (s-s_x)/(1-s_x)



#need to define the Coupling matrix J, do random generation later:

J = -0.5*np.array([[0,1,1,0,0],[0,0,1,0,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]])




# define hamiltonian

def driver_hamiltonian(t,t_max, n,target_qubit,s_x = 0.2, R = 1):  #is only for a splice in time!! would need to update the hamiltonian each time...

    #n for dimension or number of vertices!
    #give target_qubit as a number between 0 to n
    #doing it like this might be too computationally intensive - creating a new instance at each time step! vs having as variable inputs..?
    final_h = np.zeros((2**n,2**n))
    for i in range(1,n+1):   #could make this bit more efficient?
        if i == target_qubit:
           final_h += a_coefficients_true(t,t_max,s_x)*R*sigma_x(n,i)
        else:
            final_h += a_coefficients_false(t,t_max,s_x)*R*sigma_x(n,i)
    return final_h

def problem_hamiltonian(t,t_max,target_qubit,n,J,s_x=0.2):

    final_h = np.zeros((2**n,2**n))

    h = h_z(J,n)

    condition = False

    #for the first part of hamiltonian
    for i in range(1,n+1):

        if target_qubit == i:
            final_h += b_coefficients_true(t,t_max,s_x)*h[i-1]*sigma_z(n,i)
        else:
            final_h += b_coefficients_false(t,t_max,s_x)*h[i-1]*sigma_z(n,i)

    for k in range(1,n+1):

        if k == target_qubit:
            condition = True

        for j in range(k+1,n+1):

            if condition == True or j == target_qubit:   #loops have been designed such that there will be no overlap between these conditions!! 

                final_h += b_coefficients_true(t,t_max,s_x)*J[k-1,j-1]*(sigma_z(n,k)@sigma_z(n,j))

            else:

                final_h += b_coefficients_false(t,t_max,s_x)*J[k-1,j-1]*(sigma_z(n,k)@sigma_z(n,j))

        condition = False #resets condition!

    return final_h




# for the evolution 
            
def init_psi(n):    #start with this wavefunction in the evolution!

    up_state = (1/(2)**0.5)*np.array([[1],[1]])

    initial = up_state
    for i in range(n-1):  #n-1 since 1 counts as its self...
        initial = np.kron(initial,up_state)
    
    return initial





#check initial eigenvector!-------------------------------------------------------------

#check for t = 0:
t_max = 10
target_qubit = 1

n=5
initial_p_h = problem_hamiltonian(0,t_max,target_qubit,n,J)

initial_d_h = driver_hamiltonian(0,t_max,n,target_qubit)

initial_hamiltonian = initial_d_h+initial_p_h

#print(initial_d_h+initial_p_h)


def eigenvector_check(A,v,tol = 1e-6):



    Av = A@v

    ratios = Av/v

    if np.allclose(ratios, ratios[0], atol=tol, rtol=tol):
        return True, ratios[0]  # The first ratio is the eigenvalue (λ)
    return False, ratios


#inital_state = init_psi(n)
#print(eigenvector_check(initial_hamiltonian,inital_state))



# find initial eigenvector then ----------------------------------------------

num_states = 5

eigenvalues, eigenvectors = eigsh(initial_hamiltonian, k=num_states, which='SA')  #SA for smallest eigenvalues

print(eigenvalues, eigenvectors)