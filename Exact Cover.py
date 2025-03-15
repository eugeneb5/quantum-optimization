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


def unique_satisfiability_problem_generation(n,ratio=0, USA = True, satisfiability_ratio = False, save_mode = True, no_sat = False, DQA = False):

    J = np.zeros((n,n))

    B = np.zeros(n)

    criteria = False
    M=0
    attempts = 0
    if USA:
        clauses_set = set()
        while criteria == False:

            clause = np.sort(np.random.choice(np.arange(0, n), 3, replace=False))  #outputs an array e.g. [1,5,2]
            
            clause_tuple = tuple(clause)
            if clause_tuple not in clauses_set:
                clauses_set.add(clause_tuple)
            else:
                print(clause)
                print("duplicate clause made, will remove clause")
                continue
            M+=1
            print(clause)
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
                    eigenvector_solution = eigh(Hamiltonian)[1][:,0]  #first index for getting eigenvectors, second for finding the min eigenvalue corresponding eigenvector
                    # eigenvectors = eigh(Hamiltonian)[1]
                    # sln_1 = eigenvectors[:,0]
                    
                    criteria = True
                else:
                    print("number of solutions: " +str(check))
            elif no_sat:
                print("has no unique solutions")

                criteria = True
            else:
                M-=1
                print("no USA, will remove clause and repeat. Clause count: "+str(M))
                clauses_set.remove(clause_tuple)
                print("lowest eigenvalue: "+str(np.min(eigenvalues)))
            
                attempts +=1

                if attempts >=20:
                    print("terminating program, failed to find a USA")
                    return

                for index_1, element_1 in enumerate(clause):

                    B[element_1] -= 1

                    for element_2 in clause[index_1+1:3]:

                        J[element_1, element_2] -=1
        min_index = 0
        for index, element in enumerate(np.abs(eigenvector_solution)):
            if element == 1:
                print("index of the eigenvector corresponding to eigenvalue 0 of USA: " +str(index))
                min_index = index
            
        if DQA:
            return M, B, J, min_index     
        else:
            return problem_hamiltonian(M,B,J,n)
    
    
    elif satisfiability_ratio:

        M = round(n*ratio)  #might have to mess around with this
        clause_set = set()
        for i in range(M):

            duplicate = True
            while duplicate:
                clause = np.sort(np.random.choice(np.arange(0, n), 3, replace=False))
                tuple_clause = tuple(clause)

            
                if tuple_clause in clause_set:
                    print("redoing, made duplicate")

                    continue
                elif tuple_clause not in clause_set:
                    clause_set.add(tuple_clause)
                    duplicate = False
                    print("found a non-duplicate")
            print("number of clauses: "+str(i))
            print(clause)

            for index_1, element_1 in enumerate(clause):

                B[element_1] += 1

                for element_2 in clause[index_1+1:3]:

                    J[element_1, element_2] +=1

        if DQA:

            Hamiltonian = problem_hamiltonian(M,B,J,n)
            eigvalues, eigenvectors = eigh(Hamiltonian)

            minimum_eigenvalue = eigvalues[0] 
            min_eigenvector = eigenvectors[:,0]

            print("lowest eigenvalue: " +str(minimum_eigenvalue))
            
            for index, element in enumerate(np.abs(min_eigenvector)):
                if element == 1:
                    print("index of the eigenvector corresponding to minimum eigenvalue: " +str(index))
                    min_index = index

            return M,B,J,min_index
        else:
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
    print("min. eigenvalue difference is "+str(np.min(eigenvalue_difference)))

    plt.title("Example problem Hamiltonian eigenvalue Spectrum")
    for index in eigenvalue_range:
        plt.plot(x_val, eigenvalues_set[:,index], label = "eigenvalue "+str(index+1))
    plt.xlabel("Time parameter , s")
    plt.ylabel("Energy eigenvalue")
    plt.xticks([0,0.5,1])
    plt.legend()
    plt.show()

        





n = 5  #minimum of 4!
t_max = 50
q = 150

#USA = unique_satisfiability_problem_generation(n, ratio = 0.8, USA = False, satisfiability_ratio= True)
#USA= unique_satisfiability_problem_generation(n)

#Hamiltonian_spectrum(n, t_max, q, USA, number_of_eigenvalues = 4)






# to do:
#check AQA and DQA work!!  plus think about how can model with real parameters!! read the textbook bit about D-Wave
#read the 3sat paper on exponentially small gap scenario! and relate to the anders localisation paper - is that an updated version?
#read the textbook bit referencing all the different papers (page 30)
#test the algorithm given straight in the textbook
#also test the vector eigensolution and how it translates to the solution - i.e. transfer from qubits etc..
#essentially this is all to verify if our EC is right   -VERIFIED! works properly!

#then start to understand where the 'hard' problems are made - and try compare AQC with DQA!!
#maybe try something cool with the DQA if possible - like the case for multiple jumps etc - try mitigate it or encourage another one??
#like could remove the first initial jump for example - going thru it a second time if was unsuccesful - perhaps because of three jumps - and removing the initial jump therefore? a bit algorithmic...
# also try predict it - for blackbox problems?? the main idea is that for multiple problems, DQA does better than AQA if there are more instances it performs better for random problems in general!
# so if we try model it such for problems we WANT to solve - so mimic what the structures might look like - then 

#would like to find instances in EC3 where DQA doesn't work - like when the upper energy levels are all very tight with each other we can expect DQA to fail - since the spectrum becomes the sub-set of eigenvalues....

#also research applicability of EC - is it cuz it's an np complete problem that it has wide applicability!
#hence the study of it is useful!

#if want to log something like average time taken for a certain success probability - then should save each value culmulatively to a note document when runs - and can analyse each case individually too, like if there are special cases to talk about!





        

###implementing DQA ----------------------------------------

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
    
def b_coefficients_true(t,t_max,s_x):
    
    s = t/t_max
    return (s-s_x)/(1-s_x)

def b_coefficients_false(t,t_max,s_x):

    s = t/t_max
    if s<s_x:
        return 0
    else:
        return (s-s_x)/(1-s_x)
    
def driver_hamiltonian_DQA(t,t_max, target_qubit,n,B,s_x=0.2):  #is only for a splice in time!! would need to update the hamiltonian each time...

    #n for dimension or number of vertices!
    #give target_qubit as a number between 0 to n
    #doing it like this might be too computationally intensive - creating a new instance at each time step! vs having as variable inputs..?

    
    final_h = np.zeros((2**n,2**n))
    for i in range(1,n+1):   #could make this bit more efficient?
        if i == target_qubit:
           final_h += a_coefficients_true(t,t_max,s_x)*1*sigma_x(n,i)
        else:
            final_h += a_coefficients_false(t,t_max,s_x)*1*sigma_x(n,i)
    return final_h

def problem_hamiltonian_DQA(t,t_max,target_qubit,n,M,B,J,s_x=0.2):

    final_h = np.zeros((2**n,2**n))

    h = B   
    
    I = np.eye(2**n, dtype = complex)

    condition = False

    #for the first part of hamiltonian
    for i in range(1,n+1):

        if target_qubit == i:
            final_h += -0.5*b_coefficients_true(t,t_max,s_x)*(B[i-1]*sigma_z(n,i))
        else:
            final_h += -0.5*b_coefficients_false(t,t_max,s_x)*B[i-1]*sigma_z(n,i)

    for k in range(1,n+1):

        if k == target_qubit:
            condition = True

        for j in range(k+1,n+1):

            if condition == True or j == target_qubit:   #loops have been designed such that there will be no overlap between these conditions!! 

                final_h += 0.5*b_coefficients_true(t,t_max,s_x)*J[k-1,j-1]*(sigma_z(n,k)@sigma_z(n,j))  #just checked - this bit isn't working?

            else:

                final_h += 0.5*b_coefficients_false(t,t_max,s_x)*J[k-1,j-1]*(sigma_z(n,k)@sigma_z(n,j))

        condition = False #resets condition!

    return  M*I + final_h    #we haven't attached any of the conditions to M*I - since we are assuming it just moves the hamiltonian up and down anyways - so shouldn't affect anything??
        
        
def diabatic_evolution_probability_plot(target_qubit, t_max,n, M,B,J,min_index, q=150, test_superposition_state = False,test_excited_state = False):   #change r to 1e9!

    #find the initial state
    # #progress_bar = tqdm(total=num_steps, desc="Computing Eigenspectrum", position=0)
    initial_p_h = problem_hamiltonian_DQA(0,t_max,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    eigenvectors = eigh(initial_hamiltonian)[1]
    first_eigenvector = eigenvectors[:,0]
    second_eigenvector = eigenvectors[:,1]
    state = first_eigenvector

    if test_superposition_state == True:

        state = 2**(-0.5)*first_eigenvector+2**(-0.5)*second_eigenvector

    if test_excited_state == True:

        state = second_eigenvector
    
    
    
    #find the comparison state

    comparison_vector = np.zeros((2**n))
    comparison_vector[min_index] = 1



    #start the annealing process

    dt = t_max/(q)
    values = np.zeros(q+1)  #index from 0 to q array

    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)
        
        state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)
        

        probability = abs(((np.vdot(state,comparison_vector))))**2
       
        values[i] = probability


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
    

def diabatic_test_eigenspectrum(target_qubit,t_max,n,M,B,J,number_of_eigenvalues=4,q=100,r = 1, energy_difference = False):     #change r to 1e9 

    dt = t_max/(q)

    
    eigenvalues_set = np.zeros((q+1, number_of_eigenvalues))

    eigenvalue_range = np.linspace(0,number_of_eigenvalues-1, number_of_eigenvalues, dtype=int)



    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)

        eigenvalues = np.sort(eigh(Hamiltonian_at_time_instance)[0])

        for index in eigenvalue_range:

            eigenvalues_set[i,index] = eigenvalues[index]
    
    eigenvalue_difference = eigenvalues_set[:,1]-eigenvalues_set[:,0]
    x_val = np.linspace(0,1,q+1)
    zeros = np.zeros(q+1)

    

    if energy_difference == True:
        plt.title("Eigenvalue difference for annealing with target qubit " + str(target_qubit))
        plt.xlabel("Time parameter s")
        plt.ylabel("Energy difference")
        plt.yticks([0,0.5])
        plt.plot(x_val, eigenvalue_difference, label = "first minimum difference")
        plt.plot()
        plt.plot(x_val,zeros, linestyle = "--", color = "blue")

    else:
        plt.title("Hamiltonian eigenvalue spectrum for target qubit " +str(target_qubit))
        for index in eigenvalue_range:
            plt.plot(x_val, eigenvalues_set[:,index], label = "eigenvalue "+str(index+1))
        
        plt.yticks([0,1])
        
        
    plt.xticks([0,0.5,1])
    plt.legend()
    plt.show()




####TEST CASE --------------------------------



n = 6
target_qubit = 2
t_max = 50
q = 300

# M, B, J, min_index = unique_satisfiability_problem_generation(n, ratio = 0.7, USA = True, satisfiability_ratio= True, DQA = True)   #save a problem!!!! and also try change the starting hamiltonian maybe??? adapt it perhaps!? solving the decision problem only requires that the final hamiltonian is 
# np.savez("USA_values.npz", integer=M, array_1D=B, array_2D=J, index = min_index)
# print("saved")




# data = np.load("USA_values.npz")
# M = data["integer"].item()
# B = data["array_1D"]
# J = data["array_2D"]
# min_index = data["index"].item()


# H = problem_hamiltonian(M,B,J,n)

# Hamiltonian_spectrum(n, t_max, q, H, number_of_eigenvalues = 6)

# diabatic_test_eigenspectrum(target_qubit,t_max, n, M,B,J, number_of_eigenvalues=4)

# diabatic_evolution_probability_plot(target_qubit,t_max,n,M,B,J,min_index, test_excited_state=False)



#work in a saving the problem method!!




#### ---------------------- implement the E_res function 

def E_res_test(target_qubit,n,M,B,J,t_max,min_index,q=300, save_mode = False):

    initial_p_h = problem_hamiltonian_DQA(0,t_max,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    state = eigh(initial_hamiltonian)[1][:,0]
    H_problem = problem_hamiltonian(M,B,J,n)
    file_1 = "E_res_data_test_4.txt"
    file_2 = "Probability_data_test_4.txt"

    eigenvalues, eigenvectors = eigh(H_problem)
    E_0 = eigenvalues[0]
    comparison_vector = np.zeros((2**n))
    comparison_vector[min_index] = 1
    #Final_eigenvector = eigenvectors[:,0]

    print("the E_0 reference value is: "+str(E_0))
    dt = t_max/q

    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)

        state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)

    E_final = state@(H_problem@state)

    E_res = abs(E_final - E_0)  #should it be the absolute value?

    print("the E_res value for annealing time, "+str(t_max)+"seconds is: "+str(E_res))

    probability = abs(((np.vdot(state,comparison_vector))))**2

    print("probability of success is: "+str(probability))

    if save_mode:
        with open(file_1, "a") as f1, open(file_2,"a") as f2:
            f1.write(f"{E_res}\n")  # Append single value to array1.txt
            f2.write(f"{probability}\n")
        print("saved values")
        
def Plot_two_variables(filename_1 = "E_res_data_test.txt", filename_2 = "Probability_data_test.txt"):

    def load_array(filename):
        with open(filename, "r") as f:
            return np.array([float(line.strip()) for line in f if line.strip()]) 
        
    filename_1_x = "E_res_data_test.txt"
    filename_1_y = "Probability_data_test.txt"
    filename_2_x = "E_res_data_test_1.txt"
    filename_2_y = "Probability_data_test_1.txt"
    filename_3_x = "E_res_data_test_2.txt"
    filename_3_y = "Probability_data_test_2.txt"
    filename_4_x = "E_res_data_test_3.txt"
    filename_4_y = "Probability_data_test_3.txt"
    filename_5_x = "E_res_data_test_4.txt"
    filename_5_y = "Probability_data_test_4.txt"

    array_1_x = load_array(filename_1_x)
    array_1_y = load_array(filename_1_y)
    array_2_x = load_array(filename_2_x)
    array_2_y = load_array(filename_2_y)
    array_3_x = load_array(filename_3_x)
    array_3_y = load_array(filename_3_y)
    array_4_x = load_array(filename_4_x)
    array_4_y = load_array(filename_4_y)
    array_5_x = load_array(filename_5_x)
    array_5_y = load_array(filename_5_y)

    measure_from_value = 0

    plt.scatter(array_1_x[measure_from_value:], array_1_y[measure_from_value:],s=10,label = "n=8")
    plt.scatter(array_2_x[measure_from_value:], array_2_y[measure_from_value:],s=10, label = "n=7")
    plt.scatter(array_3_x[measure_from_value:], array_3_y[measure_from_value:],s=10, label = "n=6" )
    plt.scatter(array_4_x[measure_from_value:], array_4_y[measure_from_value:],s=10, label = "n=5")
    plt.scatter(array_5_x[measure_from_value:], array_5_y[measure_from_value:],s=10, label = "n=6 several problems")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()

def E_res_DQA(target_qubit,n,M,B,J, t_max_starting_value,t_max_step, save = False, q= 150):  

    #initialize the state

    initial_p_h = problem_hamiltonian_DQA(0,t_max_starting_value,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max_starting_value,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    ground_state_eigenvector = eigh(initial_hamiltonian)[1][:,0]
    H_problem = problem_hamiltonian(M,B,J,n)    
    E_res_threshold = 0.1

    #find E_0

    E_0 = eigh(H_problem)[0][0]  #should be minimum value
    print(E_0)


    #check for minimum gap size first
    #does gap size change with t_max??  NO, it doesn't change so can check min_gap

    dt = t_max_starting_value/(q)
    eigenvalue_difference = np.zeros(q+1)
    

    for i in range(0,q+1):
         h = Time_dependent_Hamiltonian(n,dt*i,t_max_starting_value,H_problem)
         instantaneous_eigenvalues_set = eigh(h)[0]
         eigenvalue_difference[i] = abs(instantaneous_eigenvalues_set[1]-instantaneous_eigenvalues_set[0])
    
    minimum_gap_size = np.min(eigenvalue_difference)
    print(minimum_gap_size)

    #initialize annealing:
    not_found = True

    t_max = t_max_starting_value

    while not_found:

        dt = t_max/(q)
        state = ground_state_eigenvector
        for i in range(0,q+1):

            Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)

            state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)

        E_final = state@(H_problem@state)

        E_res = abs(E_final - E_0)  #should it be the absolute value?

        print(E_res)


        if E_res < E_res_threshold:
            print(t_max)
            not_found = False
        else:
            t_max += t_max_step
            continue


    return minimum_gap_size, E_res



t_max_step = 1

### for generating values

# value_range = np.linspace(1,50,num=50)
# for t_m in value_range:
#     print("testing t_max value of: "+str(t_m))
#     E_res_test(target_qubit,n,M,B,J,t_m,min_index,save_mode=True)
# Plot_two_variables(filename_1="E_res_data_test_4.txt",filename_2="Probability_data_test_4.txt")

Plot_two_variables()




#E_res_test(target_qubit,n,M,B,J,t_max,min_index, save_mode=True)

# min_gap_size, E_res = E_res_DQA(target_qubit,n,M,B,J,t_max,t_max_step)

# print(min_gap_size)
# print(E_res)





        

        







    










