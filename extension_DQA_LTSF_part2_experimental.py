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

def a_coefficients_false(t,t_max,s_x):

    
    s = t/t_max
    
    if s < s_x:
        return 1
    else:
        return 1 - ((s-s_x)/(1-s_x))

def a_coefficients_true(t,t_max,s_x):    #ALTER LATER! to test how it works!!

    #test what c_x =/= 0 would be like?

    s = t/t_max

    c_x = 0

    c_1 = 0

    if s<s_x:
        return c_x
    
    else:
        return c_x +(c_1-c_x)*((s-s_x)/(1-s_x))

def h_z(J, n, kappa = 0.5, R=1e9):  #need to tinker with this one! we want to normalise to R?

    h = np.zeros(n)

    for k in range(0,n): 

        total = 0

        for j in range(0,n):

            total += -(J[k,j]+J[j,k])

        h[k] = total + kappa


    #to normalize:

    #h_norm = (h*R) / np.max(np.abs(h))

    return h

def b_coefficients_true(t,t_max,s_x):
    
    s = t/t_max
    return (s-s_x)/(1-s_x)

def b_coefficients_false(t,t_max,s_x):

    s = t/t_max
    if s<s_x:
        return 0
    else:
        return (s-s_x)/(1-s_x)

def gaussian_h_z(n,R=1e9):  #maybe remake this function - need to sample from gaussian independently??

    gaussian = np.zeros(n)
    for i in range(n):

        g_value = np.random.normal(loc = 0, scale = 1)

        gaussian[i] = g_value
    

    

    normalised_gaussian_values = gaussian*R/np.max(abs(gaussian))

    return normalised_gaussian_values




#from the milestone - for making comparison vector:

def Classical_H_ising(n,J,h_sample):


    #J is the adjacency matrix
    #need to test the three modes of h

    h = h_sample

    
    #to find H_ising now
    H = np.zeros((2**n,2**n))
    H_2 = np.zeros((2**n,2**n))
    for k in range(1,n+1):   #1 to n


        H_2 += h[k-1]*sigma_z(n,k)
        

        for j in range(k+1,n+1): #k+1 to n

            H += (J[k-1,j-1]*sigma_z(n,k)@sigma_z(n,j)) 
    
    return (H+H_2)

def Ising_search(H):

    #min_val = float('inf') #this is positive infinity
    
    val_dict = {}    

    val_dict['min_val']= float('inf')
    val_dict['index'] = None             

    for i in range(len(H)):  #H_ising will only have diagonal terms and be a perfect square matrix...

        if H[i,i] < val_dict['min_val']:

            val_dict['min_val'] = H[i,i]
            val_dict['index'] = i

    return val_dict['index']






#need to define the Coupling matrix J, do random generation later:

#J = -0.5*np.array([[0,1,1,0,0],[0,0,1,0,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0]])

J = -0.5*1e9*np.array([[0,0,1,1,0,1,1],[0,0,0,0,1,0,0],[0,0,0,1,1,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,1,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]])
n = 7
#h_sample = gaussian_h_z(n)  #need to have a method of saving it!!  write onto a txt file and save and reread it off of it!!
h_MIS = h_z(J,n)
h = 1e9*np.array([1,-0.32610452,0.16998698,-0.12109217,-0.58725647,0.19980255,-0.4370849])






# define hamiltonian

def driver_hamiltonian(t,t_max, target_qubit,n,s_x=0.2, R = 1e9):  #is only for a splice in time!! would need to update the hamiltonian each time...

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

def problem_hamiltonian(t,t_max,target_qubit,n,J,h_sample,s_x=0.2):

    final_h = np.zeros((2**n,2**n))

    h = h_sample   #needs to be the same one throughout!! that's why it didn't work before!
    
    #h = h_z(J,n)

    #h = 1e9*np.array([1,-0.32610452,0.16998698,-0.12109217,-0.58725647,0.19980255,-0.4370849])

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

                final_h += b_coefficients_true(t,t_max,s_x)*J[k-1,j-1]*(sigma_z(n,k)@sigma_z(n,j))  #just checked - this bit isn't working!!

            else:

                final_h += b_coefficients_false(t,t_max,s_x)*J[k-1,j-1]*(sigma_z(n,k)@sigma_z(n,j))

        condition = False #resets condition!

    return final_h




# for the evolution  - DEFUNCT
            
def init_psi(n):    #start with this wavefunction in the evolution! - found out is not valid!! we have a nonzero problem h term at s =0

    up_state = (1/(2)**0.5)*np.array([[1],[1]])

    initial = up_state
    for i in range(n-1):  #n-1 since 1 counts as its self...
        initial = np.kron(initial,up_state)
    
    return initial





#CHECK initial eigenvector!-------------------------------------------------------------






def eigenvector_check(A,v,tol = 1e-6):



    Av = A@v

    ratios = Av/v

    if np.allclose(ratios, ratios[0], atol=tol, rtol=tol):
        return True, ratios[0]  # The first ratio is the eigenvalue (λ)
    return False, ratios


#inital_state = init_psi(n)
#print(eigenvector_check(initial_hamiltonian,inital_state))








# find initial eigenvector then ----------------------------------------------

#num_states = 2

def is_hermitian(H, tol=1e-10):
    return np.allclose(H, H.conj().T, atol=tol)

#print(is_hermitian(initial_hamiltonian)) # it is hermitian!!
#eigenvalues, eigenvectors = eigsh(initial_hamiltonian, k=num_states, which='SA')  #SA for smallest eigenvalues; use this for larger matrices, is approximate











#find solution ------------------------------

# J = np.array([[0,1],[0,0]])
# n = 2
# h = h_z(J,n,R=1)

# J = np.array([[0,0,1,1,0,1,1],[0,0,0,0,1,0,0],[0,0,0,1,1,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,1,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]])
# n = 7
# h = h_z(J,n,R=1)


# H = Classical_H_ising(n,J,h)

# index = Ising_search(H)

#print("ground state eigenvalue index: "+str(index))

#print(H)

def is_it_diagonal(H):

    n = len(H)

    

    for i in range(n):

        for j in range(n):

            if i != j and abs(H[i][j]) > 1e-9:

                return False
    
    return True
     
#print(is_it_diagonal(H))   #so yes it is diagonal!
                
def print_diagonal_elements(H):

    n = len(H)

    for i in range(n):
        for j in range(n):

            if i ==j:
                print(H[i][j])

#print_diagonal_elements(H)

def ground_wf(n,J,h):  #is set to be a covector!  #one to be compared to!!

    H = Classical_H_ising(n,J,h) #kappa set to 0.5

    i = Ising_search(H)

    vector = np.zeros((2**n))

    vector[i] = 1

    return vector


#comparison_vector = ground_wf(n,J,h)



def check_solution_degeneracy(n,J,h):

    H = Classical_H_ising(n,J,h)

    index = Ising_search(H)

    

    check_value = H[index][index]

    degeneracy_counter = 0

    for i in range(2**n):

    

        if np.isclose(H[i][i], check_value, atol = 1e-5, equal_nan = True):
            degeneracy_counter += 1
            print(i)
            print(H[i][i])
    
    if degeneracy_counter> 1:
        return True, degeneracy_counter
    else:
        return False, degeneracy_counter


#print(check_solution_degeneracy(n,J,h))













### initialize the evolution----------------------------------------


def diabatic_evolution_test_probability(initial_eigenvector,comparison_vector,target_qubit, t_max, J, n, h_sample,q=100):   #should build in the initial_eigenvector bit later on - need to check program works first!!

    state = initial_eigenvector

    dt = t_max/q

    values = np.zeros(q+1)

    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian(i*dt,t_max,target_qubit,n,J,h_sample)+driver_hamiltonian(i*dt,t_max,target_qubit,n)
        
        state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)
        

        probability = abs(((np.vdot(state,comparison_vector))))**2
       
        values[i] = probability

    return values[q] #i.e. final value

#DONT USE THIS FUNCTION ANYMORE!!


def diabatic_test_eigenspectrum(target_qubit,t_max,J,n,h_sample,q=100,r = 1, energy_difference = False):     #change r to 1e9 

    dt = t_max/q

    ground_eig = np.zeros(q+1)
    first_eig = np.zeros(q+1)
    second_eig = np.zeros(q+1)
    min_diff_eig_1 = np.zeros(q+1)
    min_diff_eig_2 = np.zeros(q+1)
    min_diff_eig_3 = np.zeros(q+1)
    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian(i*dt,t_max,target_qubit,n,J,h_sample)+driver_hamiltonian(i*dt,t_max,target_qubit,n,R=r)

        eigenvalues, eigenvectors = eigh(Hamiltonian_at_time_instance)

        ground_eig[i] = eigenvalues[0]
        first_eig[i] = eigenvalues[1]
        second_eig[i] = eigenvalues[2]
        min_diff_eig_1[i] = eigenvalues[1]-eigenvalues[0]
        min_diff_eig_2[i] = eigenvalues[2]-eigenvalues[0]
        min_diff_eig_3[i] = eigenvalues[3] - eigenvalues[0]
    print(ground_eig[q])
    x_val = np.linspace(0,1,q+1)
    zeros = np.zeros(q+1)

    if energy_difference == True:
        plt.title("Eigenvalue difference for annealing with target qubit " + str(target_qubit))
        plt.xlabel("Time parameter s")
        plt.ylabel("Energy difference")
        plt.yticks([0,0.5])
        plt.plot(x_val, min_diff_eig_1, label = "first minimum difference")
        plt.plot(x_val, min_diff_eig_2, label = "second minimum difference")
        #plt.plot(x_val, min_diff_eig_3)
        plt.plot()
        plt.plot(x_val,zeros, linestyle = "--", color = "blue")

    else:
        plt.title("Hamiltonian eigenvalue spectrum for target qubit " +str(target_qubit))
        plt.plot(x_val,ground_eig, label= "ground state energy")
        plt.plot(x_val, first_eig, label = "first excited state energy")
        plt.plot(x_val,second_eig, label= "second excited state energy")
        plt.yticks([-5])
        
        
    plt.xticks([0,0.5,1])
    plt.legend()
    plt.show()


# target_qubit = 1
# print("this is our generated gaussian h_z"+str(h_sample))
# print("the given one" + str(h))
#diabatic_test_eigenspectrum(target_qubit,t_max,J,n,h_sample)







#this function is self contained!
def diabatic_evolution_probability_plot(target_qubit, t_max, J, n, h_sample,q=100, test_superposition_state = False,test_excited_state = False, r = 1):   #change r to 1e9!

    #find the initial state

    num_steps = q+1
    progress_bar = tqdm(total=num_steps, desc="Computing Eigenspectrum", position=0)


    initial_p_h = problem_hamiltonian(0,t_max,target_qubit,n,J,h_sample)

    initial_d_h = driver_hamiltonian(0,t_max,target_qubit,n,R=r)

    initial_hamiltonian = initial_d_h+initial_p_h

    eigenvalues, eigenvectors = eigh(initial_hamiltonian)

    first_eigenvector = eigenvectors[:,0]
    second_eigenvector = eigenvectors[:,1]

    state = first_eigenvector

    if test_superposition_state == True:

        state = 2**(-0.5)*first_eigenvector+2**(-0.5)*second_eigenvector

    if test_excited_state == True:

        state = second_eigenvector
    
    
    
    #find the comparison state

    comparison_vector = ground_wf(n,J,h_sample)




    dt = t_max/q
    values = np.zeros(q+1)  #index from 0 to q array

    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian(i*dt,t_max,target_qubit,n,J,h_sample)+driver_hamiltonian(i*dt,t_max,target_qubit,n,R=r)
        
        state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)
        

        probability = abs(((np.vdot(state,comparison_vector))))**2
       
        values[i] = probability

        progress_bar.update(1)

    x_val = np.linspace(0,1,q+1)
    p_f_line = np.zeros(q+1)
    for i in range(q+1):
        p_f_line[i]=0.99
    plt.plot(x_val, values)
    plt.plot(x_val, p_f_line, linestyle = "--")
    plt.title("Diabatic Annealing Probability curve for target qubit "+str(target_qubit)+", T = "+str(t_max)+" s")
    plt.xlabel("Time parameter, S")
    plt.ylabel("Success Probability")
    plt.xticks([0,0.5,1])
    plt.yticks([0,0.5,1])
    plt.show()
    print("for target qubit set to "+ str(target_qubit)+" we get final probability of "+str(values[q]))
    progress_bar.close()




#TEST CASE -----------------------------------------

target_qubit = 3
t_max = 45

# J = np.array([[0,0,1,1,0,1,1],[0,0,0,0,1,0,0],[0,0,0,1,1,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,1,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]])
# n = 7
# h = h_z(J,n,R=1)

J = np.array([[0,0,1,1,0,1,1],[0,0,0,0,1,0,0],[0,0,0,1,1,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,1,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]])
n = 7
h = np.array([1,-0.32610452,0.16998698,-0.12109217,-0.58725647,0.19980255,-0.4370849])
# #h = h_z(J,n)

# print(check_solution_degeneracy(n,J,h))
#diabatic_test_eigenspectrum(target_qubit, t_max, J, n, h, r=1, energy_difference=True)
#diabatic_evolution_probability_plot(target_qubit,t_max,J,n,h,test_superposition_state=False,r = 1)











#FOR generating random problems

def generate_adjacency_matrix(n):

    M = np.zeros((n,n), dtype = int)  #put dtype to ensure no floats in zero entries


    for i in range(n):
        for j in range(i+1,n):  #add i+1 to ensure they aren't self connected - can this be added though??

            edge = np.random.choice([0,1])

            M[i,j] = edge

    return M


def visualize_graph(adj_matrix):

    graph = nx.from_numpy_array(adj_matrix)

    plt.figure(figsize=(8, 8))  # Set the figure size
    nx.draw(
        graph,
        with_labels=True,            # Show node labels
        node_color='skyblue',        # Node color
        node_size=500,               # Node size
        edge_color='gray',           # Edge color
        font_size=10,                # Font size for labels
    )
    plt.title("Graph Visualization")
    plt.show()


def save_matrix(n,G,h_G, rewrite = False):

    check = check_solution_degeneracy(n,G,h_G)[0]
    if check == False:
        
        filename = "non_degenerate_solution_matrix.txt"
        print("saving non-degenerate solution graph")
    else:

        filename = "degenerate_solution_matrix.txt"
        print("saving degen_solution matrix")
    matrix = G
    if rewrite:
        with open(filename, 'w') as f:
            for row in matrix:
                row_str = ' '.join(str(num) for num in row)
                f.write(row_str + '\n')
    else:
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                for row in matrix:
                    row_str = ' '.join(str(num) for num in row)
                    f.write(row_str + '\n')
        

def read_matrix(read_degenerate):

    matrix = []
    if read_degenerate:
        filename = 'degenerate_solution_matrix.txt'
    else:
        filename = "non_degenerate_solution_matrix.txt"
    with open(filename, 'r') as file:
        for line in file:
       
            row = line.strip().split()
            row = [int(num) for num in row]
            matrix.append(row)
    return np.array(matrix)

### for timing the functions- useful for larger functions
def timed_function(func, *args):
    """Wrapper to time function execution"""
    start_time = time.time()
    result = func(*args)
    elapsed_time = time.time() - start_time
    return f"{func.__name__} completed in {elapsed_time:.2f} seconds - Result: {result}"

def unpack_and_run(f):

    return timed_function(*f)



#test case ----
#n = 9
# G = generate_adjacency_matrix(n)
# h_G = h_z(G,n,R=1)
# print(check_solution_degeneracy(n,G,h_G))
# save_matrix(n,G,h_G, rewrite = True)




if __name__ == "__main__":
    n=9
    M = read_matrix(read_degenerate=False)

    h_sample = h_z(M,n,R=1)
    target_qubit = 1
    

#visualize_graph(M)

# print(np.shape(Classical_H_ising(n,M,h_sample)))
# print(check_solution_degeneracy(n,M,h_sample))

    functions = [
        (diabatic_test_eigenspectrum, target_qubit,50, M, n, h_sample),
        (diabatic_evolution_probability_plot, target_qubit, 50,M, n, h_sample)
    ]

    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(tqdm(executor.map(unpack_and_run, functions), total=len(functions), desc="Processing"))

# diabatic_test_eigenspectrum(target_qubit,50,M,n,h_sample,r=1)
# diabatic_evolution_probability_plot(target_qubit,50,M,n,h_sample,r=1)

    for res in results:
        print(res)








#Annealing Schedule Demonstrations------------------

def annealing_schedules(t_max,s_x, a =True):

    

    #t = np.linspace(0,t_max,t_max+1)

    x_val = np.linspace(0,1,t_max+1)

    s_x_array = np.zeros(t_max+1)

    for i in range(t_max+1):

        s_x_array[i] = s_x

    y_val = np.linspace(0,1,t_max+1)
    
    if a:

        a_false = np.zeros(t_max+1)
        a_true = np.zeros(t_max+1)
        for i in range(t_max+1):

            a_false[i] = a_coefficients_false(i,t_max,s_x)
            a_true[i] = a_coefficients_true(i,t_max, s_x)


        plt.plot(x_val, a_false)

        plt.plot(x_val, a_true)

        
       
      

        

        
    else:

        b_false = np.zeros(t_max+1)
        b_true = np.zeros(t_max+1)
        
        for i in range(t_max+1):

            b_false[i] = b_coefficients_false(i,t_max,s_x)
            b_true[i] = b_coefficients_true(i,t_max,s_x)

        y_val = np.linspace(b_true[0],1,t_max+1)
        #plt.plot(x_val, b_false)
        plt.plot(x_val,b_true)
        #plt.title("for b")
       
    plt.plot(s_x_array, y_val, linestyle = "--")
    plt.xticks([0,0.5,1])
    plt.yticks([0,0.5,1])
    plt.xlabel("Time Parameter, S")
    
    plt.show()



#annealing_schedules(45,0.2,a=False)






#idea!! do comparisons between AQA (maybe use a schedule dependent on the energy gap for better prob!) and DQA, dependent on the problem's gap size - which we can alter ourselves!!, we know roughly how to do this!!












