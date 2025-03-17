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
    file_1 = "Minimum_gap_data_redo.txt"
    file_2 = "T_max_data_redo.txt"
    file_3 = "problem_dimension_redo.txt"

    if save_upper:
        file_1 = "Minimum_gap_data_upper_bound_redo.txt"   #just to fill the space, we don't really need this otherwise
        file_2 = "T_max_upper_bound_data_redo.txt"
    
    elif save_lower:
        file_1 = "Minimum_gap_data_lower_bound_redo.txt"   #just to fill the space, we don't really need this otherwise
        file_2 = "T_max_lower_bound_data_redo.txt"

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
        print(minimum_gap_size)
        print(t_max)
        with open(file_1, "a") as f1, open(file_2,"a") as f2, open(file_3, "a") as f3:
            f1.write("Test line\n")
            f1.write(f"{minimum_gap_size}\n")  # Append single value to array1.txt
            f2.write(f"{t_max}\n")
            if not save_lower and not save_upper:
                f3.write(f"{n}\n")
        print("saved values")




    return minimum_gap_size, E_res, t_max

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
t_max_starting_value = 10
t_max_step = 1
threshold_E_res = 0.0104
threshold_E_res_upper_value = 0.0112
threshold_E_res_lower_value = 0.0097  #these values are fixed/ rounded


n = 6
target_qubit_range = np.linspace(1,n,n,dtype = int)
print(target_qubit_range)
t_max_test = 100
q = 400


M, B, J, min_index = unique_satisfiability_problem_generation(n, ratio = 1.4, USA = True, satisfiability_ratio= True, DQA = True)
# np.savez("USA_values.npz", integer=M, array_1D=B, array_2D=J, index = min_index)

# data = np.load("USA_values.npz")
# M = data["integer"].item()
# B = data["array_1D"]
# J = data["array_2D"]
# min_index = data["index"].item()

print("generated random problem")

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
_,_, optimal_t_max = E_res_DQA(threshold_E_res,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode=True)

#for upperbound now
print("doing the upper bound calculation")

t_max_starting_value = round(optimal_t_max) + 5

E_res_DQA(threshold_E_res_upper_value,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode=True,save_upper=True)

#for lower bound
print("doing lower bound now")

t_max_starting_value = round(optimal_t_max) - 5

E_res_DQA(threshold_E_res_lower_value,target_qubit,n,M,B,J,t_max_starting_value,t_max_step,save_mode=True,save_lower=True)

















        

        







    










