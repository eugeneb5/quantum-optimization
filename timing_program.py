

import timeit





setup = """


from scipy.linalg import expm
import numpy as np
from functools import reduce

#----------------------------------------------


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



    #----------------------------------------------------


def H_ising(n,M,kappa):

    #M is the adjacency matrix
    h = np.zeros(n)

    for k in range(0,n): #range doesn't include n, goes from 0 to n-1 here!!

        total = 0

        for j in range(0,n):

            total += -(M[k,j]+M[j,k])

        h[k] = total + kappa

    
    #to find H_ising now
    H = np.zeros((2**n,2**n))
    H_2 = np.zeros((2**n,2**n))
    for k in range(1,n+1):   #1 to n


        H_2 += h[k-1]*sigma_z(n,k)
        

        for j in range(k+1,n+1): #k+1 to n

            H += (M[k-1,j-1]*sigma_z(n,k)@sigma_z(n,j)) 
    
    return (H+H_2)

    #----------------------------------------------


def Ising_search(H):

    min_val = float('inf') #this is positive infinity
    
    val_dict = {}    

    val_dict['min_val']= float('inf')
    val_dict['index'] = None             

    for i in range(len(H)):  #H_ising will only have diagonal terms and be a perfect square matrix...

        if H[i,i] < val_dict['min_val']:

            val_dict['min_val'] = H[i,i]
            val_dict['index'] = i

    return val_dict['index']

#-----------------------------------------

def init_psi(n):

    up_state = (1/(2)**0.5)*np.array([[1],[1]])

    initial = up_state
    for i in range(n-1):  #n-1 since 1 counts as its self...
        initial = np.kron(initial,up_state)
    
    return initial

#-------------------------------------------

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

##---------------------------------------------------------

def Hamiltonian(n,t,t_max,M):

    A = lambda t, t_max: 1 - (t/t_max)

    B = lambda t, t_max: t/t_max


    val = np.zeros((2**n,2**n))
    for i in range(1,n+1):
        val -= sigma_x(n,i)

    
    H_t = A(t,t_max)*val + B(t,t_max)*H_ising(n,M,0.5) #kappa = 0.5
    H_t = (H_t + H_t.T.conj()) / 2 
    return H_t



#----------------------------------------------

def ground_wf(n,M):  #is set to be a covector!

    H = H_ising(n,M,0.5) #kappa set to 0.5

    i = Ising_search(H)

    vector = np.zeros((2**n))

    vector[i] = 1

    return vector









def program_to_test(t_max, M, n, q=150):

    

    w_0 = ground_wf(n,M)
    
        
    prob = 0
       

    W = init_psi(n)

    dt = t_max/q

    for i in range(0,q+1):   #we have -1 to include 0 in range #original range(q,-1,-1)


            
        W = np.dot(expm(-1j*dt*Hamiltonian(n,i*dt,t_max,M)),W)  #is this matrix multiplication?

            #calculate probabilty
        
        if i == q:


            norm = abs(np.vdot(W,W))  #norm should be conserved...since unitary operator?

            

            prob = abs(((np.vdot(np.conjugate(W),w_0))))**2/norm  #this should be correct...

            
        
    return prob

M = np.array([[0,1,1,0,0],[0,0,1,0,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]])
n=5


"""

code_to_time = "program_to_test(26,M,n)"


execution_time = timeit.timeit(stmt=code_to_time, setup=setup, number=5)
print(execution_time/5)


