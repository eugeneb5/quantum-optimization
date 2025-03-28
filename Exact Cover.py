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
import csv
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.optimize import curve_fit
import math
import sys
import multiprocessing as mp
from scipy.linalg import svdvals





def sigma_z(n,j): #must be that n>=j



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


def sigma_x(n,j): #must be that n>=j

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


def Hamiltonian_spectrum(n,t_max,q,H_p,number_of_eigenvalues = 2, plot_mode = True):

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
    delta_min = np.min(eigenvalue_difference)
    print("min. eigenvalue difference is "+str(delta_min))

    if plot_mode:

        plt.title("Example problem Hamiltonian eigenvalue Spectrum")
        for index in eigenvalue_range:
            plt.plot(x_val, eigenvalues_set[:,index], label = "eigenvalue "+str(index+1))
        plt.xlabel("Time parameter , s")
        plt.ylabel("Energy eigenvalue")
        plt.xticks([0,0.5,1])
        plt.legend()
        plt.show()
    return delta_min

        
def adiabatic_probability(n,t_max,M,B,J,q=150):

    H_p = problem_hamiltonian(M,B,J,n)

    H_0 = Time_dependent_Hamiltonian(n, 0 , t_max,H_p)

    state = eigh(H_0)[1][:,0]

    min_eigenvalue = eigh(H_p)[0][0] 

    comparison_vector = eigh(H_p)[1][:,0]

    dt = t_max/(q)
    
    
    for i in range(0,q+1):

        Hamiltonian_at_time_instance = Time_dependent_Hamiltonian(n,i*dt,t_max,H_p)

        state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)

    
    final_probability = abs(((np.vdot(state,comparison_vector))))**2

    print("probability by adiabatic annealing is: "+str(final_probability))





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
           final_h += a_coefficients_true(t,t_max,s_x)*sigma_x(n,i)
        else:
            final_h += a_coefficients_false(t,t_max,s_x)*sigma_x(n,i)
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
            # print("done target_qubit "+ str(i))
        else:
            final_h += -0.5*b_coefficients_false(t,t_max,s_x)*B[i-1]*sigma_z(n,i)
            # print("done normal qubit "+str(i))

    for k in range(1,n+1):

        if k == target_qubit:
            condition = True

        for j in range(k+1,n+1):

            if condition == True and j == target_qubit:   #loops have been designed such that there will be no overlap between these conditions!!  #changed or

                final_h += 0.5*b_coefficients_true(t,t_max,s_x)*J[k-1,j-1]*(sigma_z(n,k)@sigma_z(n,j))  #just checked - this bit isn't working?

            else:

                final_h += 0.5*b_coefficients_false(t,t_max,s_x)*J[k-1,j-1]*(sigma_z(n,k)@sigma_z(n,j))

        condition = False #resets condition!

    return  M*I+ final_h    #we haven't attached any of the conditions to M*I - since we are assuming it just moves the hamiltonian up and down anyways - so shouldn't affect anything??

def init_psi_DQA(n, target_qubit, down = False):    #start with this wavefunction in the evolution!

    Hadamard_state = (1/(2)**0.5)*np.array([1,-1])
    up_state = np.array([1,0])
    down_state = np.array([0,1])
    initial = np.eye(1) # do we start off with identity??
    for i in range(1,n+1):  
        
        if i == target_qubit:
            print("target_q test")
            if down:
                initial = np.kron(initial, down_state)
            else:
                initial = np.kron(initial,up_state)
        else:
            print("generic test")
            initial = np.kron(initial, Hadamard_state)
        
    
    return initial[0]
    
def diabatic_evolution_probability_plot(target_qubit, t_max,n, M,B,J,min_index, q=150, test_superposition_state = False,test_excited_state = False,plot_mode = True):   #change r to 1e9!

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


    #TEST THIS    
    
    # state = init_psi_DQA(n,target_qubit, down = True)  #it works!! BUT whether it's down or up for the target qubit depends on B!!!! so WORKS!!
    
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
    if plot_mode:
        plt.plot(x_val, values)
        plt.plot(x_val, p_f_line, linestyle = "--")
        plt.title("Diabatic Annealing Probability curve for target qubit "+str(target_qubit)+", T = "+str(t_max)+" s")
        plt.xlabel("Time parameter, S")
        plt.ylabel("Success Probability")
        plt.xticks([0,0.5,1])
        plt.yticks([0,0.5,1])
        plt.show()
        print("for target qubit set to "+ str(target_qubit)+" we get final probability of "+str(values[q]))
    else:
        return values[q]   #this is the final probability
    
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



n = 7
target_qubit = 5
t_max = 40
q = 150

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

# adiabatic_probability(n,t_max,M,B,J)

# diabatic_test_eigenspectrum(target_qubit,t_max, n, M,B,J, number_of_eigenvalues=6)

# diabatic_evolution_probability_plot(target_qubit,t_max,n,M,B,J,min_index, test_excited_state=False)








#### ---------------------- implement the E_res function 

def E_res_test(target_qubit,n,M,B,J,t_max,min_index,q=300, save_mode = False):

    initial_p_h = problem_hamiltonian_DQA(0,t_max,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    state = eigh(initial_hamiltonian)[1][:,0]
    H_problem = problem_hamiltonian(M,B,J,n)
    file_1 = "E_res_data_test_7.txt"
    file_2 = "Probability_data_test_7.txt"

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
    filename_5_x = "E_res_data_test_5.txt"
    filename_5_y = "Probability_data_test_5.txt"
    filename_6_x = "E_res_data_test_6.txt"
    filename_6_y = "Probability_data_test_6.txt"
    filename_7_x = "E_res_data_test_7.txt"
    filename_7_y = "Probability_data_test_7.txt"
    filename_7_x = "E_res_data_test_7.txt"
    filename_7_y = "Probability_data_test_7.txt"
    filename_8_x = "adiabatic_E_res_0.txt"
    filename_8_y = "adiabatic_Prob_0.txt"


    


    array_1_x = load_array(filename_1_x)
    array_1_y = load_array(filename_1_y)
    array_2_x = load_array(filename_2_x)
    array_2_y = load_array(filename_2_y)
    array_3_x = load_array(filename_3_x)
    array_3_y = load_array(filename_3_y)
    array_4_x = load_array(filename_4_x)
    array_4_y = load_array(filename_4_y)
    array_5_x = load_array(filename_5_x) # these ones onwards are indexed 0 to 30 - they are values between 20 and 50 t_max
    array_5_y = load_array(filename_5_y)
    array_6_x = load_array(filename_6_x)
    array_6_y = load_array(filename_6_y)
    array_7_x = load_array(filename_7_x)
    array_7_y = load_array(filename_7_y)
    array_8_x = load_array(filename_8_x)
    array_8_y = load_array(filename_8_y)

    measure_from_value = 30
 
    x_data = np.concatenate((array_1_x[measure_from_value:],array_2_x[measure_from_value:], array_3_x[measure_from_value:], array_4_x[measure_from_value:], array_5_x[(measure_from_value-30):],array_6_x[(measure_from_value-30):],array_7_x[(measure_from_value-25):], array_8_x))

    y_data = np.concatenate((array_1_y[measure_from_value:],array_2_y[measure_from_value:],array_3_y[measure_from_value:],array_4_y[measure_from_value:],array_5_y[(measure_from_value-30):],array_6_y[(measure_from_value-30):],array_7_y[(measure_from_value-25):],array_8_y))

    poly_coeffs, cov_matrix = np.polyfit(x_data, y_data, 1, cov = True)  # Degree 1 for linear
    poly_fit = np.poly1d(poly_coeffs)

    m_uncertainty = np.sqrt(cov_matrix[0, 0])
    c_uncertainty = np.sqrt(cov_matrix[1, 1])
    
    initial_E_res = np.max(x_data)
    final_E_res = np.min(x_data)

    x_val = np.linspace(initial_E_res,final_E_res,50)

    y_val = poly_fit(x_val)

    plt.plot(x_val, y_val)
    
    plt.scatter(array_1_x[measure_from_value:], array_1_y[measure_from_value:],s=10,label = "n=8")
    plt.scatter(array_2_x[measure_from_value:], array_2_y[measure_from_value:],s=10, label = "n=7")
    plt.scatter(array_3_x[measure_from_value:], array_3_y[measure_from_value:],s=10, label = "n=6" )
    plt.scatter(array_4_x[measure_from_value:], array_4_y[measure_from_value:],s=10, label = "n=5")
    plt.scatter(array_5_x[(measure_from_value-30):], array_5_y[(measure_from_value-30):],s=10, label = "n=9")   #this one is fine as long as measure_from_value >= 30!!
    plt.scatter(array_6_x[(measure_from_value-30):], array_6_y[(measure_from_value-30):],s=10, label = "n=7")  #removed this? since is slow?
    plt.scatter(array_7_x[(measure_from_value-30):], array_7_y[(measure_from_value-30):],s=10, label = "n=7")
    plt.scatter(array_8_x[measure_from_value:], array_8_y[measure_from_value:],s=10, label = "adiabatic n=7")  #adiabatic
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()

    return poly_coeffs, m_uncertainty, c_uncertainty   #returns an array

def round_to_1_sf(value):

    if value == 0:
        return 0
    else:
        order = math.floor(math.log10(value))
        rounded = round(value, -order)

        return rounded

def E_res_DQA(E_res_threshold,target_qubit,n,M,B,J, t_max_starting_value,t_max_step, save_mode = False, q= 400, save_upper = False, save_lower = False):  

    #initialize the state

    initial_p_h = problem_hamiltonian_DQA(0,t_max_starting_value,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max_starting_value,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    ground_state_eigenvector = eigh(initial_hamiltonian)[1][:,0]
    H_problem = problem_hamiltonian(M,B,J,n)    
    # file_1 = "Minimum_gap_data_redo.txt"
    file_2 = "norm_test_T_max.txt"
    # file_3 = "problem_dimension_redo.txt"

    if save_upper:
        # file_1 = "Minimum_gap_data_upper_bound_redo.txt"   #just to fill the space, we don't really need this otherwise
        file_2 = "norm_test_T_max_U.txt"
    
    elif save_lower:
        # file_1 = "Minimum_gap_data_lower_bound_redo.txt"   #just to fill the space, we don't really need this otherwise
        file_2 = "norm_test_T_max_L.txt"

    #find E_0



    E_0 = eigh(H_problem)[0][0]  #should be minimum value
    print(E_0)


    #check for minimum gap size first
    #does gap size change with t_max??  NO, it doesn't change so can check min_gap

    dt = t_max_starting_value/(q)
    # eigenvalue_difference = np.zeros(q+1)
    

    # for i in range(0,q+1):
    #      h = Time_dependent_Hamiltonian(n,dt*i,t_max_starting_value,H_problem)
    #      instantaneous_eigenvalues_set = eigh(h)[0]
    #      eigenvalue_difference[i] = abs(instantaneous_eigenvalues_set[1]-instantaneous_eigenvalues_set[0])
    
    # minimum_gap_size = np.min(eigenvalue_difference)
    # print("minimum gap size is: "+str(minimum_gap_size))

    
    #initialize annealing:
    not_found = True

    t_max = t_max_starting_value

    previous_value  = 10000
    previous_t_max = 1
    bisection = False
    pp_t_max = 1
    difference_E_res = 0

    while not_found:

        dt = t_max/(q)
        state = ground_state_eigenvector
        for i in range(0,q+1):

            Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)

            state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)

        E_final = state@(H_problem@state)

        E_res = abs(E_final - E_0)  #should it be the absolute value?

        print("testing for t_max, "+str(t_max)+", gives E_res of: "+str(E_res))

        

        if np.isclose(E_res,E_res_threshold, rtol = 0.001):   #how accurate could we get it?

            print("the critical t_max value is: "+str(t_max)+" with residual energy of: "+str(E_res))
            # print("the corresponding minimum gap size is "+str(minimum_gap_size))
            not_found = False
            continue


        elif E_res > 7.5*E_res_threshold and not bisection:   #do we try factor of 10 or 100?
            print("still far")
            pp_t_max = previous_t_max
            previous_value = E_res
            #print("updated prev: "+str(previous_value))
            previous_t_max = t_max
            t_max += 10*t_max_step
            print("new t_max value is " +str(t_max))
            continue

        elif E_res > E_res_threshold and not bisection:
            pp_t_max = previous_t_max
            previous_value = E_res
            #print("updated prev: "+str(previous_value))
            previous_t_max = t_max
            print("getting close")
            t_max += 2*t_max_step
            print("new t_max value is " +str(t_max))
            continue
        
        
        
        elif E_res-E_res_threshold <0 and not bisection:   #i.e. sign change detected since overstepped FIRST TIME !
            print("entering bisection period")
            difference_E_res = E_res - E_res_threshold
            pp_t_max = previous_t_max
            t_max_holder = t_max   #save current t_max before updating it
            t_max = (previous_t_max+t_max)/2
            previous_t_max = t_max_holder #then update prev_t_max once new t_max has been calculated
            previous_value = E_res
            bisection = True
            
            continue
    

        elif bisection and (E_res - E_res_threshold )*difference_E_res<0:
            difference_E_res = E_res - E_res_threshold
            pp_t_max = previous_t_max
            t_max_holder = t_max
            t_max = (previous_t_max+t_max)/2
            previous_t_max = t_max_holder
            continue
        elif bisection and (E_res - E_res_threshold)*difference_E_res >0:
            difference_E_res = E_res - E_res_threshold
            
            previous_t_max = t_max
            t_max = (pp_t_max+t_max)/2
            
            continue

        
    if save_mode:
       
        # with open(file_1, "a") as f1, open(file_2,"a") as f2, open(file_3, "a") as f3:
        #     f1.write(f"{minimum_gap_size}\n")  # Append single value to array1.txt
        #     f2.write(f"{t_max}\n")
        #     if not save_lower and not save_upper:
        #         f3.write(f"{n}\n")
        # print("saved values")
        with open(file_2, "a") as f2:

            f2.write(f"{t_max}\n")
        print("saved t_max value")





    return t_max

def E_res_test_adiabatic(n,M,B,J,t_max,num,min_index,q=400,save_mode = False):

    H_problem = problem_hamiltonian(M,B,J,n)
    initial_hamiltonian = Time_dependent_Hamiltonian(n,0,t_max,H_problem)
    state = eigh(initial_hamiltonian)[1][:,0]
    

    file_1 = f"adiabatic_E_res_{num}.txt"
    file_2 = f"adiabatic_Prob_{num}.txt"

    eigenvalues, eigenvectors = eigh(H_problem)
    E_0 = eigenvalues[0]
    comparison_vector = np.zeros((2**n))
    comparison_vector[min_index] = 1

    print("the E_0 reference value is: "+str(E_0))
    dt = t_max/q
    for i in range(0,q+1):

        Hamiltonian_at_time_instance = Time_dependent_Hamiltonian(n,i*dt,t_max,H_problem)

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

def update_counter(filename, condition_met):
    try:
        # Read current counter from file
        with open(filename, "r") as file:
            count = int(file.read().strip())
    except FileNotFoundError:
        # If file doesn't exist, start at 0
        count = 0

    # If result == x (condition is True), increment counter
    if condition_met:
        count += 1

    # Write updated counter back to file
    with open(filename, "w") as file:
        file.write(str(count))

    print(f"Current count: {count}")
    return count

def log_integer(filename, value):
    with open(filename, 'a') as f:
        f.write(f"{value}\n")

def E_res_AQA(E_res_threshold,n,M,B,J, t_max_starting_value,t_max_step, save_mode = False, q= 1000, save_upper = False, save_lower = False):  

    #initialize the state

    H_problem = problem_hamiltonian(M,B,J,n)

    H_0 = Time_dependent_Hamiltonian(n, 0 , t_max_starting_value,H_problem)

    ground_state = eigh(H_0)[1][:,0]  #initial state

    E_0 = eigh(H_problem)[0][0]  #true minimum value

   
    file_1 = "Minimum_gap_data_adiabatic.txt"
    file_2 = "T_max_data_adiabatic.txt"
    file_3 = "problem_dimension_adiabatic.txt"

    if save_upper:
        file_1 = "Minimum_gap_data_upper_bound_adiabatic.txt"   #just to fill the space, we don't really need this otherwise
        file_2 = "T_max_upper_bound_data_adiabatic.txt"
    
    elif save_lower:
        file_1 = "Minimum_gap_data_lower_bound_adiabatic.txt"   #just to fill the space, we don't really need this otherwise
        file_2 = "T_max_lower_bound_data_adiabatic.txt"

    #find E_0



   


    #check for minimum gap size first
    #does gap size change with t_max??  NO, it doesn't change so can check min_gap

    dt = t_max_starting_value/(q)
    eigenvalue_difference = np.zeros(q+1)
    

    for i in range(0,q+1):
         h = Time_dependent_Hamiltonian(n,dt*i,t_max_starting_value,H_problem)
         instantaneous_eigenvalues_set = eigh(h)[0]
         eigenvalue_difference[i] = abs(instantaneous_eigenvalues_set[1]-instantaneous_eigenvalues_set[0])
    
    minimum_gap_size = np.min(eigenvalue_difference)
    print("minimum gap size is: "+str(minimum_gap_size))

    #initialize annealing:
    not_found = True

    t_max = t_max_starting_value

    previous_value  = 10000
    previous_t_max = 1
    bisection = False
    pp_t_max = 1
    difference_E_res = 0

    while not_found:

        dt = t_max/(q)

        state = ground_state  #crucial to reset state!
        
        for i in range(0,q+1):

            Hamiltonian_at_time_instance = Time_dependent_Hamiltonian(n,dt*i,t_max,H_problem)

            state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)

        E_final = state@(H_problem@state)

        E_res = abs(E_final - E_0)  #should it be the absolute value?

        print("testing for t_max, "+str(t_max)+", gives E_res of: "+str(E_res))

        

        if np.isclose(E_res,E_res_threshold, rtol = 0.001):   #how accurate could we get it?

            print("the critical t_max value is: "+str(t_max)+" with residual energy of: "+str(E_res))
            print("the corresponding minimum gap size is "+str(minimum_gap_size))
            not_found = False
            continue


        elif E_res > 7.5*E_res_threshold and not bisection:   #do we try factor of 10 or 100?
            print("still far")
            pp_t_max = previous_t_max
            previous_value = E_res
            #print("updated prev: "+str(previous_value))
            previous_t_max = t_max
            t_max += 20*t_max_step
            print("new t_max value is " +str(t_max))
            continue

        elif E_res > E_res_threshold and not bisection:
            pp_t_max = previous_t_max
            previous_value = E_res
            #print("updated prev: "+str(previous_value))
            previous_t_max = t_max
            print("getting close")
            t_max += 2*t_max_step
            print("new t_max value is " +str(t_max))
            continue
        
        
        
        elif E_res-E_res_threshold <0 and not bisection:   #i.e. sign change detected since overstepped FIRST TIME !
            print("entering bisection period")
            difference_E_res = E_res - E_res_threshold
            pp_t_max = previous_t_max
            t_max_holder = t_max   #save current t_max before updating it
            t_max = (previous_t_max+t_max)/2
            previous_t_max = t_max_holder #then update prev_t_max once new t_max has been calculated
            previous_value = E_res
            bisection = True
            
            continue
    

        elif bisection and (E_res - E_res_threshold )*difference_E_res<0:
            difference_E_res = E_res - E_res_threshold
            pp_t_max = previous_t_max
            t_max_holder = t_max
            t_max = (previous_t_max+t_max)/2
            previous_t_max = t_max_holder
            continue
        elif bisection and (E_res - E_res_threshold)*difference_E_res >0:
            difference_E_res = E_res - E_res_threshold
            
            previous_t_max = t_max
            t_max = (pp_t_max+t_max)/2
            
            continue

        
    if save_mode:
        
        with open(file_1, "a") as f1, open(file_2,"a") as f2, open(file_3, "a") as f3:
            f1.write(f"{minimum_gap_size}\n")  # Append single value to array1.txt
            f2.write(f"{t_max}\n")
            if not save_lower and not save_upper:
                f3.write(f"{n}\n")
        print("saved values")

    return t_max
# E_res_test(target_qubit,n,M,B,J,t_max,min_index,q=200)




### for generating E_res vs prob plot--------------------------------------------

# value_range = np.linspace(151,200,num=50)


# for t_m in value_range:
#     print("testing t_max value of: "+str(t_m))
#     E_res_test_adiabatic(n,M,B,J,t_m,0,min_index,save_mode=True)





######for calculating E residual threshold...


# polycoeff_array, m_uncert, c_uncert = Plot_two_variables()

# print(polycoeff_array) # first is gradient, second is intercept
# m = polycoeff_array[0]  #m is negative
# c = polycoeff_array[1]  

# #uncertainties are given >0:

# m_shallower = m+ m_uncert
# m_steeper = m - m_uncert

# c_lower = c - c_uncert
# c_upper = c+c_uncert



# threshold_E_res = (0.99-c)/m  # we want the one 

# threshold_E_res_upper_value = (0.99-c_upper)/m_shallower
# threshold_E_res_lower_value = (0.99-c_lower)/m_steeper


# print("mean value: "+str(threshold_E_res))  #we are using this new value now!

# print("upper limit: "+str(threshold_E_res_upper_value))
# print("lower limit: "+str(threshold_E_res_lower_value))

# threshold_E_res_lower_value = 0.009650497725576529
# threshold_E_res_upper_value = 0.01119135713847231

############--------------------------------------------------------------------------

def run(n,rerun = False, save = True):
    t_max_starting_value = 1
    t_max_step = 10
    threshold_E_res = 0.0104
    threshold_E_res_upper_value = 0.0112
    threshold_E_res_lower_value = 0.0097  #these values are fixed/ rounded


    
    target_qubit_range = np.linspace(1,n,n,dtype = int)
    print(target_qubit_range)
    t_max_test = 100
    q = 400

    if not rerun:
        M, B, J, min_index = unique_satisfiability_problem_generation(n, ratio = 1.4, USA = True, satisfiability_ratio= True, DQA = True)
        np.savez("USA_values.npz", integer=M, array_1D=B, array_2D=J, index = min_index)
        print("generated random problem")

    if rerun:
        data = np.load("USA_values.npz")
        M = data["integer"].item()
        B = data["array_1D"]
        J = data["array_2D"]
        min_index = data["index"].item()
        print("reloaded previous problem")

    

    H = problem_hamiltonian(M,B,J,n)

    # Hamiltonian_spectrum(n, t_max, q, H, number_of_eigenvalues = 6)

    ######----------------------checking degeneracy

    eigenvalues = eigh(H)[0]

    min_eigenvalue = np.min(eigenvalues)

    degeneracy = 0

    for i in eigenvalues:

        if i == min_eigenvalue:
            degeneracy += 1


    print("random problem degeneracy is "+str(degeneracy))


    ####-----------------------------------------------find successful target_qubit


    incompatible_problem= True
    index_target_qubit = 0
    fail = False

    while incompatible_problem:
        if index_target_qubit > n:
            print("unsuccessful problem, quitting program")
            fail = True
            update_counter("number_of_times_failed.txt",fail)
            log_integer("failed_problems_ratio.txt",M)
            sys.exit()
        target_qubit = target_qubit_range[index_target_qubit]
        print("testing with qubit "+str(target_qubit))
        final_probability = diabatic_evolution_probability_plot(target_qubit, t_max_test, n,M,B,J,min_index, plot_mode = False)
        if final_probability > 1/(degeneracy+1) or np.isclose(final_probability, (1/(degeneracy+1))):

            print("found successful problem with target qubit "+str(target_qubit)+"with success probability of: "+str(final_probability))

            incompatible_problem = False
            continue
        else:
            index_target_qubit += 1

    ###------------------------------------------now find optimal t_max



    target_qubit = target_qubit_range[index_target_qubit]
    # diabatic_test_eigenspectrum(target_qubit,t_max_test,n,M,B,J,number_of_eigenvalues=6,q=q)

    print("now finding optimal t_max for threshold E_res")
    _,_, optimal_t_max = E_res_DQA(threshold_E_res,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode = save)

    #for upperbound now
    print("doing the upper bound calculation")

    t_max_starting_value = round(optimal_t_max) 

    E_res_DQA(threshold_E_res_upper_value,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode = save,save_upper=True)

    #for lower bound
    print("doing lower bound now")

    t_max_starting_value = round(optimal_t_max) 

    E_res_DQA(threshold_E_res_lower_value,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode= save,save_lower=True)



    #####then initialise for AQA!----------------------------------------------------------------------------------------



    t_max_starting_value = 1  #need to redefine here again
    print("doing AQA simulation now")

    optimal_t_max = E_res_AQA(threshold_E_res,n,M,B,J,t_max_starting_value,t_max_step=10,save_mode=save)

    t_max_starting_value = round(optimal_t_max) 

    E_res_AQA(threshold_E_res_upper_value,n,M,B,J,t_max_starting_value,t_max_step=10,save_mode=save,save_upper = True)   #for upperbound E_res

    E_res_AQA(threshold_E_res_lower_value,n,M,B,J,t_max_starting_value,t_max_step=10,save_mode=save, save_lower = True)  #lowerbound


# run(5, rerun = True, save = False)

# for i in range(4):

#     print("running instance "+str(i))
    
#     run(8)




#check if is saving properly...it might be doing the same thing again!
# DONT EDIT txt files while the program is running!


#checking results ----------------------------------------------







def plot_graph_adiabatic():
    
    def load_array(filename):
        with open(filename, "r") as f:
            return np.array([float(line.strip()) for line in f if line.strip()]) 

    x_val_adiabatic = load_array("T_max_data_adiabatic.txt")
    y_val_adiabatic = load_array("Minimum_gap_data_adiabatic.txt")

    x_upper_adiabatic = load_array("T_max_lower_bound_data_adiabatic.txt")
    x_lower_adiabatic = load_array("T_max_upper_bound_data_adiabatic.txt")


    x_val_diabatic = load_array("T_max_data_redo.txt")
    y_val_diabatic = load_array("Minimum_gap_data_redo.txt")

    x_upper_diabatic = load_array("T_max_lower_bound_data_redo.txt")
    x_lower_diabatic = load_array("T_max_upper_bound_data_redo.txt")


    #for adiabatic first:
    n = len(x_val_adiabatic)
    x_err_adiabatic = np.zeros((n,2))
    for i in range(n):

        x_err_adiabatic[i,0] = abs(x_val_adiabatic[i] - x_lower_adiabatic[i])  #0 index is lower
        x_err_adiabatic[i,1] = abs(x_val_adiabatic[i] - x_lower_adiabatic[i])    #1 index is higher
    
    #for diabatic:
    n = len(x_val_diabatic)
    x_err_diabatic = np.zeros((n,2))
    for i in range(n):

        x_err_diabatic[i,0] = abs(x_val_diabatic[i] - x_lower_diabatic[i])  #0 index is lower
        x_err_diabatic[i,1] = abs(x_val_diabatic[i] - x_lower_diabatic[i]) 
    plt.errorbar(x_val_adiabatic, y_val_adiabatic, xerr=[x_err_adiabatic[:,0],x_err_adiabatic[:,1]], fmt='o', capsize=2, label = "AQA", markersize = 4)
    plt.errorbar(x_val_diabatic, y_val_diabatic,xerr=[x_err_diabatic[:,0],x_err_diabatic[:,1]], fmt='o', capsize=2, label = "DQA", markersize = 4)
    plt.legend()
    plt.show()

# plot_graph_adiabatic()

#plot both datasets for adiabatic and diabatic on the same graph ? and show very clearly the difference...

        


def max_error_estimation_AQA(k,t_max,n,M,B,J):

    delta_t = t_max/k

    H_p = problem_hamiltonian(M,B,J,n)

    H = np.zeros((2**n,2**n), dtype = complex)

    I = np.eye(2**n, dtype = complex)

    for i in range(0,k+1):  #we want to include k - so will have k+1 total instances!!

        H_instance = Time_dependent_Hamiltonian(n,i*delta_t,t_max,H_p)

        H += H_instance


    average_err = (k+1)*(expm(-1j*H*t_max/(k+1)**2)- I + (1j*H*t_max/(k+1)**2))   #double check this - might be - !!

    return np.max(eigh(average_err)[0])  #i.e. finding the maximum expectation value! 


def max_error_estimation_DQA(k,t_max,n,M,B,J,target_qubit):  #target_qubit shouldn't matter?

    dt = t_max/k

    H = np.zeros((2**n,2**n), dtype = complex)

    I = np.eye(2**n, dtype = complex)

    for i in range(0,k+1):  #we want to include k - so will have k+1 total instances!!

        H_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B) -M*I #this would only produce error for after 

        H += H_instance


    average_err = (k+1)*(expm(-1j*H*t_max/(k+1)**2)- I + (1j*H*t_max/(k+1)**2))   #double check this - might be - !!

    return np.max(eigh(average_err)[0])  #i.e. finding the maximum expectation value! 





#lets measure the point where there are phase transitions?--------------------------------------------------
    


def phase_transition(target_qubit,t_max,n,M,B,J,q=100,r=1,first_excited_state = False):

    initial_p_h = problem_hamiltonian_DQA(0,t_max,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    eigenvectors = eigh(initial_hamiltonian)[1]
    first_eigenvector = eigenvectors[:,0]
   


      #start the annealing process

    dt = t_max/(q)
    average_magnetisation_values = np.zeros((q+1,n), dtype = complex)  #index from 0 to q array

    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)
        
        # state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)
        
        # for qubit in range(1,n+1):  #we want to access indexes 0 to n-1

        #     average_magnetisation = state@(sigma_z(n,qubit)@state)   #should be between -1 and 1!!? check this...

        #     average_magnetisation_values[i,qubit-1] = average_magnetisation
        
        

        if first_excited_state:
            eigenvector =  eigh(Hamiltonian_at_time_instance)[1][:,1]
        else:
            eigenvector = eigh(Hamiltonian_at_time_instance)[1][:,0]


        for qubit in range(1,n+1):  #we want to access indexes 0 to n-1

             average_magnetisation = eigenvector@(sigma_z(n,qubit)@eigenvector)   #should be between -1 and 1!!? check this...

             average_magnetisation_values[i,qubit-1] = average_magnetisation


    x_val = np.linspace(0,1,q+1)

    for qubit in range(n):

        # if not qubit == target_qubit-1:

        plt.plot(x_val, average_magnetisation_values[:,qubit], label = "qubit "+str(qubit+1))
    # plt.plot(x_val, average_magnetisation_values[:,target_qubit-1])
    plt.legend()
    plt.show()

def phase_transition_AQA(t_max,n,M,B,J,q=100,r=1,first_excited_state = False):

    problem_H = problem_hamiltonian(M,B,J,n)
    # initial_hamiltonian = Time_dependent_Hamiltonian(n,0,t_max,)

    # eigenvectors = eigh(initial_hamiltonian)[1]
    # first_eigenvector = eigenvectors[:,0]
   


      #start the annealing process

    dt = t_max/(q)
    average_magnetisation_values = np.zeros((q+1,n), dtype = complex)  #index from 0 to q array

    for i in range(0,q+1):

        Hamiltonian_at_time_instance = Time_dependent_Hamiltonian(n,i*dt,t_max,problem_H)
        
        # state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)
        
        # for qubit in range(1,n+1):  #we want to access indexes 0 to n-1

        #     average_magnetisation = state@(sigma_z(n,qubit)@state)   #should be between -1 and 1!!? check this...

        #     average_magnetisation_values[i,qubit-1] = average_magnetisation
        
        

        if first_excited_state:
            eigenvector =  eigh(Hamiltonian_at_time_instance)[1][:,1]
        else:
            eigenvector = eigh(Hamiltonian_at_time_instance)[1][:,0]


        for qubit in range(1,n+1):  #we want to access indexes 0 to n-1

             average_magnetisation = eigenvector@(sigma_z(n,qubit)@eigenvector)   #should be between -1 and 1!!? check this...

             average_magnetisation_values[i,qubit-1] = average_magnetisation


    x_val = np.linspace(0,1,q+1)

    for qubit in range(n):

        # if not qubit == target_qubit-1:

        plt.plot(x_val, average_magnetisation_values[:,qubit], label = "qubit "+str(qubit+1))
    # plt.plot(x_val, average_magnetisation_values[:,target_qubit-1])
    plt.legend()
    plt.show()

def level_crossing_finder(target_qubit, n, M, B, J, q=1000):


    dt = t_max/(q)
    # delta = np.zeros(q+1, dtype = complex)  #index from 0 to q array
    
    inner_product_array = np.zeros(q+1, dtype = complex)  #0 to q
    # previous_ground_eigenvector = np.zeros(2**n, dtype = complex)

    initial_hamiltonian = problem_hamiltonian_DQA(0,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(0,t_max,target_qubit,n,B)

    previous_state = eigh(initial_hamiltonian)[1][:,0]


    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)

        eigenvalues,eigenvectors = eigh(Hamiltonian_at_time_instance)

        ground = eigenvectors[:,0]
        first = eigenvectors[:,1]

        ground_eig = eigenvalues[0]
        first_eig = eigenvalues[1]

        comparison = (np.vdot(previous_state, ground))**2

        delta = np.abs(first_eig-ground_eig)

        if np.isclose(delta, 0):  #do we add tolerance ourselves??

            print("found level crossing at s= "+str((i*dt)/t_max))


        previous_state = ground

        # H_perturbation = driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)
        # transition_probability = np.abs(first@(H_perturbation@ground))**2  #because of small numerical errors?? we don't get perfect...? e.g. in eig? maybe try a 

        inner_product_array[i] = comparison

    x_val = np.linspace(0,1,q+1)

    plt.plot(x_val, inner_product_array)
    plt.show()

def init_psi_DQA(n, target_qubit, down = False):    #start with this wavefunction in the evolution!

    Hadamard_state = (1/(2)**0.5)*np.array([1,-1])
    up_state = np.array([1,0])
    down_state = np.array([0,1])
    initial = np.eye(1) # do we start off with identity??
    for i in range(1,n+1):  
        
        if i == target_qubit:
            print("target_q test")
            if down:
                initial = np.kron(initial, down_state)
            else:
                initial = np.kron(initial,up_state)
        else:
            print("generic test")
            initial = np.kron(initial, Hadamard_state)
        
    
    return initial[0]

def init_psi(n):    #start with this wavefunction in the evolution!

    up_state = (1/(2)**0.5)*np.array([1,1])

    initial = up_state
    for i in range(n-1):  #n-1 since 1 counts as its self...
        initial = np.kron(initial,up_state)
    
    return initial


# also want to check to find out the bit arrangement at those points in time? can we do it? its not going to be perfect 1s and zeros?
# might have to do a method of bisection....! to get the crossing points, precisely... will be difficult since delta even with absolute will never go down to zero... would need to consider the first derivative...


#redo the initial state thing - try make it!!! ourselves!! with target qubit in mind!! and check orthogonality etc!!

        
# n = 7
# target_qubit = 5
# t_max = 100


# M, B, J, min_index = unique_satisfiability_problem_generation(n, ratio = 0.7, USA = True, satisfiability_ratio= True, DQA = True)   #save a problem!!!! and also try change the starting hamiltonian maybe??? adapt it perhaps!? solving the decision problem only requires that the final hamiltonian is 

# np.savez("USA_values.npz", integer=M, array_1D=B, array_2D=J, index = min_index)
# print("saved")




# data = np.load("USA_values.npz")
# M = data["integer"].item()
# B = data["array_1D"]
# J = data["array_2D"]
# min_index = data["index"].item()

# H_p = problem_hamiltonian(M,B,J,n)

# Hamiltonian_spectrum(n,t_max,150,H_p)
# diabatic_test_eigenspectrum(target_qubit,t_max, n, M,B,J, number_of_eigenvalues=6)
# level_crossing_finder(target_qubit,n,M,B,J)
# phase_transition(target_qubit,t_max,n,M,B,J, first_excited_state= False)



# H = driver_hamiltonian_DQA(0,t_max,target_qubit,n,B) + problem_hamiltonian_DQA(0,t_max, target_qubit,n,M,B,J,s_x = 0.5)



# state_1 = init_psi_DQA(n, target_qubit,down = True) #down
# state_2 = init_psi_DQA(n,target_qubit)  #up
# state_3 = init_psi(n) #control test




# eig_1 = np.real(np.round(np.vdot((H@state_1),state_1)/np.linalg.norm(state_1)**2))
# eig_2 = np.real(np.round(np.vdot((H@state_2),state_2)/np.linalg.norm(state_2)**2))

# inner_product = np.vdot(state_1,state_2)

# print("for down state: "+str(eig_1))
# print("for up state: " +str(eig_2))
# print("inner product is "+str(inner_product))  #SHOWS THEYRE ORTHOGONAL!!!


# eig = eigh(H)[0][0]

# print("directly off of H_initial, : "+str(eig))




        
# diabatic_evolution_probability_plot(target_qubit,t_max,n,M,B,J,min_index)






#to do:

#try matching the first order QPT points to level crossings on the H spectrum
# test for the convergence of the second first order QPT point as c_x value is changed - from adiabatic case? DO WE NEED TO DO THIS??... isolate that bit maybe? and ignore the starting trivial QPT...?

#check the first and ground states and the orthogonality of the states as a function of s?  also just check the 0001111 etc state and locate the target qubit bit?? is this possible?? i.e we translate the eigenvector into a binary??

#IT WORKS!! and could we check what the states are maybe?? maybe not....might just have to keep to this pattern of orthogonality!! but makes sense now!!
#compare to AQA example maybe?
#at points where inner product goes to zero - check!
#THINK OF AS TENSOR PRODUCTS?? tensor products of many different superpositions (i.e alpha*up +beta*down etc...) - except one is -1 and other is 1 and these are orthogonal?

#how could we fit in idea of spin flips? check the paper that does it with q?

#AND THE IDEA OF FIRST ORDER PHASE TRANSITIONS FITS IN REALLY NICELY WITH THIS!! we dont get the second first order QPT sometimes because of the criterion mentioned in the paper not being met?? try check this!!

#we may not need to quantatively apply the idea - but can see the curvature of the one above is visibly less than the one below in cases where no frustration occurs!
    









#### testing specific case of interest!!!--------------------------------------------------------------

# clauses = np.array([[0,4,5],[0,4,6],[0,2,4],[2,4,6],[0,4,7],[0,2,7],[0,1,3],[3,4,7]])  #this is a particular degenerate case - understand why we're unable to solve?? in theory the E_res should go towards zero regardless...?


# n = 8
# target_qubit = 5
# t_max = 50
# q=500

def problem_generation_from_clauses(n,clauses):
    M = len(clauses)
    J = np.zeros((n,n))

    B = np.zeros(n)

    for i in range(M):

        for index_1, element_1 in enumerate(clauses[i]):

            B[element_1] += 1

            for element_2 in clauses[i][index_1+1:3]:
               

                J[element_1, element_2] +=1

    H = problem_hamiltonian(M,B,J,n)

    min_eigenvector = eigh(H)[1][:,0]

    min_index = 0

    for index, element in enumerate(np.abs(min_eigenvector)):

        if element == 1:
        
            min_index = index
            # print(min_index)  #so 71 and 72 

    return M,B,J,min_index


# M,B,J,min_index = problem_generation_from_clauses(n,clauses)
# # Hamiltonian_spectrum(n,t_max,q,H)

# # diabatic_evolution_probability_plot(target_qubit,t_max,n,M,B,J,min_index)

# # E_res_DQA(0.014,target_qubit,n,M,B,J,10,2,q=200)   #doesn't work - it alternates between the two value bounds...explain why maybe???

# diabatic_test_eigenspectrum(target_qubit,t_max,n,M,B,J)




n = 9
t_max = 100
t_max_starting_value = 10
t_max_step = 10
not_found = True
target_qubit = 3
threshold_E_res = 0.0104

# while not_found:
#     M, B, J, min_index = unique_satisfiability_problem_generation(n, USA = True, DQA = True)   #save a problem!!!! and also try change the starting hamiltonian maybe??? adapt it perhaps!? solving the decision problem only requires that the final hamiltonian is 

#     np.savez("USA_values.npz", integer=M, array_1D=B, array_2D=J, index = min_index)
#     print("saved")

#     probability = diabatic_evolution_probability_plot(target_qubit,t_max,n,M,B,J,min_index, plot_mode=False)

#     if probability <= 0.98 and probability >= 0.4:
#         print(probability)
#         print("found problem")
#         not_found = False

#     else:
#         print(probability)
#         print("havent found, repeating")
#         continue


#     H = problem_hamiltonian(M,B,J,n)

#     delta_min = Hamiltonian_spectrum(n,t_max,q,H, plot_mode = False)


#     if delta_min <= 0.185:
#         print(delta_min)
#         print("found a small gap problem!")
#         not_found = False
#     else:
#         print(delta_min)
#         print("failed, doing again")
#         continue








###set of problem clauses that are interesting...

clauses_1 = np.array([[1,3,6],[3,4,6],[0,2,7],[1,3,7],[1,3,4],[0,3,5]])  #only for target qubit 3 does it produce the interesting results, for 4 it's normal... n=8

clauses_2 = np.array([[0,1,5],[1,3,7],[0,3,6],[2,3,6],[4,5,6],[1,3,4],[0,6,7]])  #small min gap but fast DQA time, n=8

clauses_3 = np.array([[4,5,6],[0,1,7],[0,4,7],[1,2,5],[3,6,7],[2,6,7]])  #t_max = 92, min_gap = 0.129  , which target_qubit? n=8

clauses_4 = np.array([[2,3,6],[1,3,4],[1,6,7],[1,4,7],[1,2,4],[3,5,6],[0,1,5]]) #n=8

clauses_5 = np.array([[1,6,8],[1,4,7],[1,5,6],[1,2,4],[1,3,6],[0,1,3],[4,5,8]]) #target_qubit 1 - long DQA time, n=9

clauses_6 = np.array([[1,5,7],[4,5,6],[3,4,5],[5,7,8],[0,1,6],[1,2,7],[3,6,7]])  #target_qubti 1 - long DQA time, n=9

clauses_7 = np.array([[0,3,4],[0,3,5],[1,4,5],[0,1,4],[0,2,4]]) #n=6, M=5, visual false positive!!!! VERY GOOD COUNTER EXAMPLE... #target qubit 3!!!!

clauses_8 = np.array([[0,4,5],[0,3,5],[0,1,5],[2,3,4],[1,3,5]]) #n=6, target qubit = 3, t_max = 84! and does show that small ish gap thing...

#MOST IMPORTANT - there seems to be a lot of variation, but within this variation, it is always the case that DQA triumphs!! but why...
#test lower clause limits..?

# n=8

# M,B,J,min_index = problem_generation_from_clauses(n,clauses_1)



# n=10

n=8
q = 10000
t_max_AQA = 400
t_max = 100
target_qubit = 3

# M, B, J, min_index = unique_satisfiability_problem_generation(n, USA = True, DQA = True)   #save a problem!!!! and also try change the starting hamiltonian maybe??? adapt it perhaps!? solving the decision problem only requires that the final hamiltonian is 

# np.savez("USA_values.npz", integer=M, array_1D=B, array_2D=J, index = min_index)
# print("saved")

# data = np.load("USA_values.npz")
# M = data["integer"].item()
# B = data["array_1D"]
# J = data["array_2D"]
# min_index = data["index"].item()


#we are doing for AQA first....


# q_param = np.linspace(100,10000,100).astype(int)
# t_param = np.linspace(10,400,100).astype(int)

# # print(q_param)

# n_q = 100
# n_t = 100

# print(max_error_estimation_AQA(q,t_max_AQA,n,M,B,J))

# q_param = np.linspace(100,10000,5).astype(int)
# t_param = np.linspace(10,400,10).astype(int)

# n_q = 10
# n_t = 10



# error_results_AQA = np.zeros((n_q,n_t))

# for index_q, q in enumerate(q_param):  #y
#     print(index_q)

#     for index_t, t in enumerate(t_param):  #x

#         err = max_error_estimation_AQA(q,t,n,M,B,J)

#         if err > 1:
#             err = 1
#         error_results_AQA[index_q,index_t] = err**2

# plt.figure(figsize=(8, 6))
# plt.imshow(error_results_AQA, origin='lower', cmap='RdBu_r', aspect='auto')
# plt.colorbar()
# # plt.xlabel('t_max')
# # plt.ylabel('q')

# x_ticks = [0, len(t_param) // 2, len(t_param) - 1]
# y_ticks = [0, len(q_param) // 2, len(q_param) - 1]

# # Optional: set tick labels to actual param values
# plt.xticks(ticks=x_ticks, labels=t_param[x_ticks])
# plt.yticks(ticks=y_ticks, labels=q_param[y_ticks])


# plt.savefig('my_plot.png')
# plt.show()





# AQA_error = max_error_estimation_AQA(q,t_max_AQA,n,M,B,J)

# DQA_error = max_error_estimation_DQA(q,t_max,n,M,B,J,target_qubit)

# print(AQA_error)
# print(DQA_error)









# H = problem_hamiltonian(M,B,J,n)

# Hamiltonian_spectrum(n,t_max,q,H,number_of_eigenvalues=4)

# phase_transition_AQA(t_max,n,M,B,J)



# diabatic_test_eigenspectrum(target_qubit,t_max,n,M,B,J)

# # E_res_DQA(threshold_E_res,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,q=200)
# # # E_res_AQA(threshold_E_res,n,M,B,J,t_max_starting_value,t_max_step)

# phase_transition(target_qubit,t_max,n,M,B,J, first_excited_state=False)








####finding the second order phase transitions...

def susceptibiity_graph(target_qubit,t_max,n,M,B,J,AQA =False,q=100,r=1,first_excited_state = False, second_deriv = False):

    initial_p_h = problem_hamiltonian_DQA(0,t_max,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    eigenvectors = eigh(initial_hamiltonian)[1]
    first_eigenvector = eigenvectors[:,0]
   


      #start the annealing process

    dt = t_max/(q)
    average_magnetisation_values = np.zeros((q+1,n), dtype = complex)  #index from 0 to q array

    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)
        
        # state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)
        
        # for qubit in range(1,n+1):  #we want to access indexes 0 to n-1

        #     average_magnetisation = state@(sigma_z(n,qubit)@state)   #should be between -1 and 1!!? check this...

        #     average_magnetisation_values[i,qubit-1] = average_magnetisation
        
        

        if first_excited_state:
            eigenvector =  eigh(Hamiltonian_at_time_instance)[1][:,1]
        else:
            eigenvector = eigh(Hamiltonian_at_time_instance)[1][:,0]


        for qubit in range(1,n+1):  #we want to access indexes 0 to n-1

             average_magnetisation = eigenvector@(sigma_z(n,qubit)@eigenvector)   #should be between -1 and 1!!? check this...

             average_magnetisation_values[i,qubit-1] = average_magnetisation


    x_val = np.linspace(0,1,q+1)

    delta_x = x_val[1]-x_val[0]

    susceptibility_values = np.zeros((q+1, n), dtype = complex)

    Xi_deriv_values = np.zeros((q+1,n),dtype = complex)

    for qubit in range(n): #0 to n-1

        susceptibility_values[:,qubit] = np.gradient(average_magnetisation_values[:,qubit],delta_x)

        if second_deriv:

            Xi_deriv_values[:,qubit] = np.gradient(susceptibility_values[:,qubit],delta_x)

            plt.plot(x_val, Xi_deriv_values[:,qubit], label = "for qubit "+str(qubit+1))
        else:

            plt.plot(x_val, susceptibility_values[:,qubit], label = "for qubit "+str(qubit+1))

    plt.ylim(-10,10)
    plt.show()

   
    



    # plt.legend()
    # plt.show()



def susceptibiity_graph(target_qubit,t_max,n,M,B,J,AQA =False,q=100,r=1,first_excited_state = False, second_deriv = False):

    initial_p_h = problem_hamiltonian_DQA(0,t_max,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    eigenvectors = eigh(initial_hamiltonian)[1]
    first_eigenvector = eigenvectors[:,0]
   


      #start the annealing process

    dt = t_max/(q)
    average_magnetisation_values = np.zeros((q+1,n), dtype = complex)  #index from 0 to q array

    for i in range(0,q+1):

        Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)
        
        # state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)
        
        # for qubit in range(1,n+1):  #we want to access indexes 0 to n-1

        #     average_magnetisation = state@(sigma_z(n,qubit)@state)   #should be between -1 and 1!!? check this...

        #     average_magnetisation_values[i,qubit-1] = average_magnetisation
        
        

        if first_excited_state:
            eigenvector =  eigh(Hamiltonian_at_time_instance)[1][:,1]
        else:
            eigenvector = eigh(Hamiltonian_at_time_instance)[1][:,0]


        for qubit in range(1,n+1):  #we want to access indexes 0 to n-1

             average_magnetisation = eigenvector@(sigma_z(n,qubit)@eigenvector)   #should be between -1 and 1!!? check this...

             average_magnetisation_values[i,qubit-1] = average_magnetisation


    x_val = np.linspace(0,1,q+1)

    delta_x = x_val[1]-x_val[0]

    susceptibility_values = np.zeros((q+1, n), dtype = complex)

    Xi_deriv_values = np.zeros((q+1,n),dtype = complex)

    for qubit in range(n): #0 to n-1

        susceptibility_values[:,qubit] = np.gradient(average_magnetisation_values[:,qubit],delta_x)

        if second_deriv:

            Xi_deriv_values[:,qubit] = np.gradient(susceptibility_values[:,qubit],delta_x)

            plt.plot(x_val, Xi_deriv_values[:,qubit], label = "for qubit "+str(qubit+1))
        else:

            plt.plot(x_val, susceptibility_values[:,qubit], label = "for qubit "+str(qubit+1))

    plt.ylim(-10,10)
    plt.show()


# susceptibiity_graph(target_qubit,t_max,n,M,B,J, first_excited_state=False, second_deriv=False)



###lets do a ratio comparison graph for solving 


def E_res_DQA_ratio_test(E_res_threshold,target_qubit,n,M,B,J, t_max_starting_value,t_max_step, save_mode = False, q= 400, save_upper = False, save_lower = False):  

    #initialize the state

    initial_p_h = problem_hamiltonian_DQA(0,t_max_starting_value,target_qubit,n,M,B,J)
    initial_d_h = driver_hamiltonian_DQA(0,t_max_starting_value,target_qubit,n,B)
    initial_hamiltonian = initial_d_h+initial_p_h
    ground_state_eigenvector = eigh(initial_hamiltonian)[1][:,0]
    H_problem = problem_hamiltonian(M,B,J,n)    
    file_1 = "problem_dimension_ratio_test.txt"
    file_2 = "T_max_data_ratio_test.txt"
    # file_3 = "problem_dimension_redo.txt"

    if save_upper:
        file_2 = "T_max_data_ratio_test_upper_bound.txt"  #just to fill the space, we don't really need this otherwise
        
    
    elif save_lower:
          #just to fill the space, we don't really need this otherwise
        file_2 = "T_max_data_ratio_test_lower_bound.txt" 

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
    print("minimum gap size is: "+str(minimum_gap_size))

    #initialize annealing:
    not_found = True

    t_max = t_max_starting_value

    previous_value  = 10000
    previous_t_max = 1
    bisection = False
    pp_t_max = 1
    difference_E_res = 0

    while not_found:

        dt = t_max/(q)
        state = ground_state_eigenvector
        for i in range(0,q+1):

            Hamiltonian_at_time_instance = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)

            state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)

        E_final = state@(H_problem@state)

        E_res = abs(E_final - E_0)  #should it be the absolute value?

        print("testing for t_max, "+str(t_max)+", gives E_res of: "+str(E_res))

        

        if np.isclose(E_res,E_res_threshold, rtol = 0.001):   #how accurate could we get it?

            print("the critical t_max value is: "+str(t_max)+" with residual energy of: "+str(E_res))
            print("the corresponding minimum gap size is "+str(minimum_gap_size))
            not_found = False
            continue


        elif E_res > 7.5*E_res_threshold and not bisection:   #do we try factor of 10 or 100?
            print("still far")
            pp_t_max = previous_t_max
            previous_value = E_res
            #print("updated prev: "+str(previous_value))
            previous_t_max = t_max
            t_max += 10*t_max_step
            print("new t_max value is " +str(t_max))
            continue

        elif E_res > E_res_threshold and not bisection:
            pp_t_max = previous_t_max
            previous_value = E_res
            #print("updated prev: "+str(previous_value))
            previous_t_max = t_max
            print("getting close")
            t_max += 2*t_max_step
            print("new t_max value is " +str(t_max))
            continue
        
        
        
        elif E_res-E_res_threshold <0 and not bisection:   #i.e. sign change detected since overstepped FIRST TIME !
            print("entering bisection period")
            difference_E_res = E_res - E_res_threshold
            pp_t_max = previous_t_max
            t_max_holder = t_max   #save current t_max before updating it
            t_max = (previous_t_max+t_max)/2
            previous_t_max = t_max_holder #then update prev_t_max once new t_max has been calculated
            previous_value = E_res
            bisection = True
            
            continue
    

        elif bisection and (E_res - E_res_threshold )*difference_E_res<0:
            difference_E_res = E_res - E_res_threshold
            pp_t_max = previous_t_max
            t_max_holder = t_max
            t_max = (previous_t_max+t_max)/2
            previous_t_max = t_max_holder
            continue
        elif bisection and (E_res - E_res_threshold)*difference_E_res >0:
            difference_E_res = E_res - E_res_threshold
            
            previous_t_max = t_max
            t_max = (pp_t_max+t_max)/2
            
            continue

        
    if save_mode:
       
        with open(file_1, "a") as f1, open(file_2,"a") as f2:
            f2.write(f"{t_max}\n")
            if not save_lower and not save_upper:
                f1.write(f"{n}\n")
        print("saved values")

    return t_max





def E_res_AQA_ratio_test(E_res_threshold,n,M,B,J, t_max_starting_value,t_max_step, save_mode = False, q= 1000, save_upper = False, save_lower = False):  

    #initialize the state

    H_problem = problem_hamiltonian(M,B,J,n)

    H_0 = Time_dependent_Hamiltonian(n, 0 , t_max_starting_value,H_problem)

    ground_state = eigh(H_0)[1][:,0]  #initial state

    E_0 = eigh(H_problem)[0][0]  #true minimum value

   
    file_1 = "problem_dim_ratio_test_adiabatic.txt"
    file_2 = "T_max_data_adiabatic_ratio_test.txt"
   

    if save_upper:
          
        file_2 = "T_max_data_adiabatic_ratio_test_upper_bound.txt"
    
    elif save_lower:
        
        file_2 = "T_max_data_adiabatic_ratio_test_lower_bound.txt"

    #find E_0



   


    #check for minimum gap size first
    #does gap size change with t_max??  NO, it doesn't change so can check min_gap

    dt = t_max_starting_value/(q)
    eigenvalue_difference = np.zeros(q+1)
    

    for i in range(0,q+1):
         h = Time_dependent_Hamiltonian(n,dt*i,t_max_starting_value,H_problem)
         instantaneous_eigenvalues_set = eigh(h)[0]
         eigenvalue_difference[i] = abs(instantaneous_eigenvalues_set[1]-instantaneous_eigenvalues_set[0])
    
    minimum_gap_size = np.min(eigenvalue_difference)
    print("minimum gap size is: "+str(minimum_gap_size))

    #initialize annealing:
    not_found = True

    t_max = t_max_starting_value

    previous_value  = 10000
    previous_t_max = 1
    bisection = False
    pp_t_max = 1
    difference_E_res = 0

    while not_found:

        dt = t_max/(q)

        state = ground_state  #crucial to reset state!
        
        for i in range(0,q+1):

            Hamiltonian_at_time_instance = Time_dependent_Hamiltonian(n,dt*i,t_max,H_problem)

            state = np.dot(expm(-1j*dt*Hamiltonian_at_time_instance),state)

        E_final = state@(H_problem@state)

        E_res = abs(E_final - E_0)  #should it be the absolute value?

        print("testing for t_max, "+str(t_max)+", gives E_res of: "+str(E_res))

        

        if np.isclose(E_res,E_res_threshold, rtol = 0.001):   #how accurate could we get it?

            print("the critical t_max value is: "+str(t_max)+" with residual energy of: "+str(E_res))
            print("the corresponding minimum gap size is "+str(minimum_gap_size))
            not_found = False
            continue


        elif E_res > 7.5*E_res_threshold and not bisection:   #do we try factor of 10 or 100?
            print("still far")
            pp_t_max = previous_t_max
            previous_value = E_res
            #print("updated prev: "+str(previous_value))
            previous_t_max = t_max
            t_max += 20*t_max_step
            print("new t_max value is " +str(t_max))
            continue

        elif E_res > E_res_threshold and not bisection:
            pp_t_max = previous_t_max
            previous_value = E_res
            #print("updated prev: "+str(previous_value))
            previous_t_max = t_max
            print("getting close")
            t_max += 2*t_max_step
            print("new t_max value is " +str(t_max))
            continue
        
        
        
        elif E_res-E_res_threshold <0 and not bisection:   #i.e. sign change detected since overstepped FIRST TIME !
            print("entering bisection period")
            difference_E_res = E_res - E_res_threshold
            pp_t_max = previous_t_max
            t_max_holder = t_max   #save current t_max before updating it
            t_max = (previous_t_max+t_max)/2
            previous_t_max = t_max_holder #then update prev_t_max once new t_max has been calculated
            previous_value = E_res
            bisection = True
            
            continue
    

        elif bisection and (E_res - E_res_threshold )*difference_E_res<0:
            difference_E_res = E_res - E_res_threshold
            pp_t_max = previous_t_max
            t_max_holder = t_max
            t_max = (previous_t_max+t_max)/2
            previous_t_max = t_max_holder
            continue
        elif bisection and (E_res - E_res_threshold)*difference_E_res >0:
            difference_E_res = E_res - E_res_threshold
            
            previous_t_max = t_max
            t_max = (pp_t_max+t_max)/2
            
            continue

        
    if save_mode:
        
        with open(file_1, "a") as f1, open(file_2,"a") as f2:
            f2.write(f"{t_max}\n")
            if not save_lower and not save_upper:
                f1.write(f"{n}\n")
        print("saved values")

    return t_max





def run_ratio(n,rerun = False, save = True):
    t_max_starting_value = 1
    t_max_step = 10
    threshold_E_res = 0.0104
    threshold_E_res_upper_value = 0.0112
    threshold_E_res_lower_value = 0.0097  #these values are fixed/ rounded


    
    target_qubit_range = np.linspace(1,n,n,dtype = int)
    print(target_qubit_range)
    t_max_test = 100
    q = 400

    if not rerun:
        M, B, J, min_index = unique_satisfiability_problem_generation(n, ratio = 1.4, USA = True, satisfiability_ratio= True, DQA = True)
        np.savez("USA_values.npz", integer=M, array_1D=B, array_2D=J, index = min_index)
        print("generated random problem")

    if rerun:
        data = np.load("USA_values.npz")
        M = data["integer"].item()
        B = data["array_1D"]
        J = data["array_2D"]
        min_index = data["index"].item()
        print("reloaded previous problem")

    

    H = problem_hamiltonian(M,B,J,n)

    # Hamiltonian_spectrum(n, t_max, q, H, number_of_eigenvalues = 6)

    ######----------------------checking degeneracy

    eigenvalues = eigh(H)[0]

    min_eigenvalue = np.min(eigenvalues)

    degeneracy = 0

    for i in eigenvalues:

        if i == min_eigenvalue:
            degeneracy += 1


    print("random problem degeneracy is "+str(degeneracy))


    ####-----------------------------------------------find successful target_qubit


    incompatible_problem= True
    index_target_qubit = 0
    fail = False

    while incompatible_problem:
        if index_target_qubit > n:
            print("unsuccessful problem, quitting program")
            fail = True
            update_counter("number_of_times_failed.txt",fail)
            log_integer("failed_problems_ratio.txt",M)
            sys.exit()
        target_qubit = target_qubit_range[index_target_qubit]
        print("testing with qubit "+str(target_qubit))
        final_probability = diabatic_evolution_probability_plot(target_qubit, t_max_test, n,M,B,J,min_index, plot_mode = False)
        if final_probability > 1/(degeneracy+1) or np.isclose(final_probability, (1/(degeneracy+1))):

            print("found successful problem with target qubit "+str(target_qubit)+"with success probability of: "+str(final_probability))

            incompatible_problem = False
            continue
        else:
            index_target_qubit += 1

    ###------------------------------------------now find optimal t_max



    target_qubit = target_qubit_range[index_target_qubit]
    # diabatic_test_eigenspectrum(target_qubit,t_max_test,n,M,B,J,number_of_eigenvalues=6,q=q)

    print("now finding optimal t_max for threshold E_res")
    E_res_DQA_ratio_test(threshold_E_res,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode = save)

    #for upperbound now
    # print("doing the upper bound calculation")

    # t_max_starting_value = round(optimal_t_max) 

    # E_res_DQA_ratio_test(threshold_E_res_upper_value,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode = save,save_upper=True)

    #for lower bound
    # print("doing lower bound now")

    # t_max_starting_value = round(optimal_t_max) 

    # E_res_DQA_ratio_test(threshold_E_res_lower_value,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode= save,save_lower=True)



    #####then initialise for AQA!----------------------------------------------------------------------------------------



    t_max_starting_value = 1  #need to redefine here again
    print("doing AQA simulation now")

    E_res_AQA_ratio_test(threshold_E_res,n,M,B,J,t_max_starting_value,t_max_step=10,save_mode=save)

    # t_max_starting_value = round(optimal_t_max) 

    # E_res_AQA_ratio_test(threshold_E_res_upper_value,n,M,B,J,t_max_starting_value,t_max_step=10,save_mode=save,save_upper = True)   #for upperbound E_res

    # E_res_AQA_ratio_test(threshold_E_res_lower_value,n,M,B,J,t_max_starting_value,t_max_step=10,save_mode=save, save_lower = True)  #lowerbound





# for i in range(10):
    
#     run_ratio(9)
    
    



#after, try testing for degenerate cases - make sure not to save etc... - we want to add since we've considered E_Res!!

#there's a limit on what we can do to begin with....





#test for different inhomogenous field values!!! but this would require redefinitions of starting states...
#when is 1/B, doesn't work well... over 400
#when is B, works similar or better? maybe doesn't work better? we can check this... 210 -220 ish
#then for just one -about the same, 210...

#show that the phase transitions don't change!!

#changing B changes position of phase transitions, but not the fact that phase transitions don't happen?!

#is it worth exploring as a potential fix?





####measure change in Hamiltonian, normed!



def spectral_norm_DQA(n,t_max,target_qubit,M,B,J,q=10,plot=True):

    dt = t_max/q

    norm_values = np.zeros(q)

    for i in range(0,q):  #i.e. going from 0 to q-1

        Hamiltonian_before = problem_hamiltonian_DQA(i*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA(i*dt,t_max,target_qubit,n,B)
        Hamiltonian_after = problem_hamiltonian_DQA((i+1)*dt,t_max,target_qubit,n,M,B,J)+driver_hamiltonian_DQA((i+1)*dt,t_max,target_qubit,n,B)  #so final one will be q

        A = (Hamiltonian_after - Hamiltonian_before)/dt   #rough derivative definition...so higher q, more precise this will be...

        svals = svdvals(A)
        norm_values[i] = svals[0]

    if plot:

        x_val = np.linspace(0,1,q)

        plt.plot(x_val,norm_values)
        plt.show()

    return np.max(norm_values)

    
def spectral_norm_AQA(n,t_max,M,B,J,q=10):

    dt = t_max/q

    H_p = problem_hamiltonian(M,B,J,n)

    norm_values = np.zeros(q)

    for i in range(0,q):  #i.e. going from 0 to q-1

        Hamiltonian_before = Time_dependent_Hamiltonian(n,i*dt,t_max,H_p)
        Hamiltonian_after = Time_dependent_Hamiltonian(n,(i+1)*dt,t_max,H_p) #so final one will be q

        A = (Hamiltonian_after - Hamiltonian_before)/dt   #rough derivative definition...so higher q, more precise this will be...

        svals = svdvals(A)
        norm_values[i] = svals[0]

    x_val = np.linspace(0,1,q)

    plt.plot(x_val,norm_values)
    plt.show()

    return np.max(norm_values)






# max_norm = spectral_norm_DQA(n,t_max,target_qubit,M,B,J,q=10)
# max_norm_2 = spectral_norm_AQA(n,t_max,M,B,J,q=10)

# print(max_norm)
# print(max_norm_2)





#is it worth measuring then the variation in spectral norm with n? NO - we can just do spectral norm with T_max! for DQA!!
#it might be worth verifying a rough 
#but fact we see a very clean 1/x^2 relation for AQA in other graph - means min gap size is the 'stronger', more limiting condition - which confirms O(N) vs O(>exp(N)) comp time dependecnies!!!


def run_norm_DQA(n,rerun = False, save = True):
    t_max_starting_value = 1
    t_max_step = 10
    threshold_E_res = 0.0104
    threshold_E_res_upper_value = 0.0112
    threshold_E_res_lower_value = 0.0097  #these values are fixed/ rounded


    
    target_qubit_range = np.linspace(1,n,n,dtype = int)
    print(target_qubit_range)
    t_max_test = 100
    q = 400

    if not rerun:
        M, B, J, min_index = unique_satisfiability_problem_generation(n, ratio = 1.4, USA = True, satisfiability_ratio= True, DQA = True)
        np.savez("USA_values.npz", integer=M, array_1D=B, array_2D=J, index = min_index)
        print("generated random problem")

    if rerun:
        data = np.load("USA_values.npz")
        M = data["integer"].item()
        B = data["array_1D"]
        J = data["array_2D"]
        min_index = data["index"].item()
        print("reloaded previous problem")

    

    H = problem_hamiltonian(M,B,J,n)

    # Hamiltonian_spectrum(n, t_max, q, H, number_of_eigenvalues = 6)

    ######----------------------checking degeneracy

    eigenvalues = eigh(H)[0]

    min_eigenvalue = np.min(eigenvalues)

    degeneracy = 0

    for i in eigenvalues:

        if i == min_eigenvalue:
            degeneracy += 1


    print("random problem degeneracy is "+str(degeneracy))


    ####-----------------------------------------------find successful target_qubit


    incompatible_problem= True
    index_target_qubit = 0
    fail = False

    while incompatible_problem:
        if index_target_qubit > n:
            print("unsuccessful problem, quitting program")
            fail = True
            update_counter("number_of_times_failed.txt",fail)
            log_integer("failed_problems_ratio.txt",M)
            sys.exit()
        target_qubit = target_qubit_range[index_target_qubit]
        print("testing with qubit "+str(target_qubit))
        final_probability = diabatic_evolution_probability_plot(target_qubit, t_max_test, n,M,B,J,min_index, plot_mode = False)
        if final_probability > 1/(degeneracy+1) or np.isclose(final_probability, (1/(degeneracy+1))):

            print("found successful problem with target qubit "+str(target_qubit)+"with success probability of: "+str(final_probability))

            incompatible_problem = False
            continue
        else:
            index_target_qubit += 1

    ###------------------------------------------now find optimal t_max

    

    target_qubit = target_qubit_range[index_target_qubit]
    # diabatic_test_eigenspectrum(target_qubit,t_max_test,n,M,B,J,number_of_eigenvalues=6,q=q)

    print("now finding optimal t_max for threshold E_res")
    optimal_t_max = E_res_DQA(threshold_E_res,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode = save)

    #for upperbound now
    print("doing the upper bound calculation")

    t_max_starting_value = round(optimal_t_max) 

    E_res_DQA(threshold_E_res_upper_value,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode = save,save_upper=True)

    #for lower bound
    print("doing lower bound now")

    t_max_starting_value = round(optimal_t_max) 

    E_res_DQA(threshold_E_res_lower_value,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode= save,save_lower=True)


    #now to save norm value:

    norm = spectral_norm_DQA(n,optimal_t_max,target_qubit,M,B,J,q=10,plot=False)
    print("the norm is "+str(norm))
    print("with M " +str(M) )
    print("for dimension, n "+str(n))

    file_name_1 = "norm_data_norm.txt"
    file_name_2 = "norm_data_M.txt"
    file_name_3 = "norm_data_n.txt"
    if save:

        with open(file_name_1, "a") as f1, open(file_name_2, "a") as f2, open(file_name_3,"a") as f3 :

                f1.write(f"{norm}\n")
                f2.write(f"{M}\n")
                f3.write(f"{n}\n")
        print("saved norm,M,n values")



    #####then initialise for AQA!----------------------------------------------------------------------------------------



   

for i in range(10):

    run_norm_DQA(6, save=True)

    run_norm_DQA(7, save=True)












#####