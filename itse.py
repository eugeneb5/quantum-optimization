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

def s(psi, H, t):
    return -H(t)*psi

def s_time_independent(psi,H):

    return -H*psi


def RK4(dim, total_time, n,psi_0,H, is_time_dependent):  #n is number of iterations

    dt = total_time/n

    #ts = np.arrange(total_time)*dt

    t = 0

    ys = np.zeros((n,2**dim))

    y = psi_0

    def dpsi_dt (psi, H, t):

        if is_time_dependent == True:

            return -1*np.dot(H(t),psi)  #this is for imaginary time!!
        else:

            return -1*np.dot(H,psi)
        


    for i in range(n):

        ys[i] = y

        

        k_0 = dt*(dpsi_dt(y,H,t))
        k_1 = dt*(dpsi_dt(y+(k_0/2), H, t+(dt/2)))
        k_2 = dt*(dpsi_dt(y+(k_1/2), H, t+(dt/2)))
        k_3 = dt*(dpsi_dt(y+k_2 , H, t+dt))

        y = y + (1/6)*(k_0+2*k_1+2*k_2+k_3)

        y = y/np.linalg.norm(y)  #normalize psi at each time step

        t += dt


    return ys


def psi_initial(n):

    return np.random.uniform(low = 1e-10, high = 1, size = 2**n)  #low is exclusive, high is inclusive




# initialize test for time-independent H:
H = H_ising(5,M,k)

psi_0 = psi_initial(5)




r = RK4(5,50,1000,psi_0,H,is_time_dependent=False)

print(r.shape)

print(r[999,18])

#print(len(r))


#what should i graph?
# try E_res first!

def graph_E_res(H_classic, values, known_eigenvalue):  #requires knowledge of what eigenvalue is! for n=5 case, min value is -6.5


    n = len(values)

    x_val = np.linspace(0,n,n)

    y_val = np.zeros(n)

    for i in range(n):

        psi = values[i]

        E_actual = np.dot(psi.conj().T, np.dot(H_classic,psi))

        y_val[i] = E_actual - known_eigenvalue

    plt.plot(x_val, y_val)


    plt.show()


#check for this basic example...

graph_E_res(H, r, -6.5)


#now we try for time dependent adiabatic theorem




    

        















