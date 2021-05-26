import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
import pandas as pd
import math
# from random import uniform
# from random import sample
import random

# 2-0.108-0.39


global global_m
global global_c


def clt(Mean, Total):
	N = Total
	Variance = Mean*(1-Mean)
	print(f'\nVariance = {Variance}')
	clt_lower = Mean - 2*(math.sqrt(Variance/N))
	clt_upper = Mean + 2*(math.sqrt(Variance/N))
	return clt_lower, clt_upper

def partA_fraction(N, P, S):
	Fraction_with_no_Symptoms = (P-S)/N
	Fraction_with_Symptoms = S/N
	Fraction_Negative = (N-P)/N
	return Fraction_with_no_Symptoms, Fraction_with_Symptoms, Fraction_Negative




#TANMAY DATASET
dataset = pd.read_csv('./final2021.csv', header=None)
N, P, S = 10695, 1477, 895


#Q1 Part A
print("\nQ1 Part A")
Fraction_with_no_Symptoms, Fraction_with_Symptoms, Fraction_Negative = partA_fraction(N,P,S) 
print(f'Fraction of People tested +ve without symptoms = {Fraction_with_no_Symptoms}')
print(f'Fraction of People tested +ve with symptoms = {Fraction_with_Symptoms}')

#mu = (0*Fraction_Negative)+(1*Fraction_with_no_Symptoms) + (2*Fraction_with_Symptoms)
#print(f'Mean mu = {mu}')

#Q1 Part B 
print("\nQ1 Part B")
clt_lower, clt_upper = clt(Fraction_with_no_Symptoms, N)
print(f'CI(tested +ve but have no symptoms) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')

clt_lower, clt_upper = clt(Fraction_with_Symptoms, N)
print(f'CI(tested +ve and have symptoms) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')


# Q1 Part D
print("\nQ1 Part D")
False_Positive = 0.01
False_Negative = 0.1

Fraction_with_no_Symptoms, Fraction_with_Symptoms, Fraction_Negative = partA_fraction(N, P, S) 

People_Tested_Positive_without_symptoms = Fraction_with_no_Symptoms * N
Real_Positive = ( (1-False_Positive)*People_Tested_Positive_without_symptoms) + (False_Negative*(N-P))
Fraction_Real_Positive = Real_Positive/N
print(f'Real Positive = {Real_Positive} and Fraction of population which are truely positve but have no symptoms = {Fraction_Real_Positive}')


#Q1 Part E 
print("\nQ1 Part E")
clt_lower, clt_upper = clt(Fraction_Real_Positive, N)
print(f'CI(Real Positive People) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')



# Q1 Part F
print("\nQ1 Part F")
#P(A|B) = P(B|A)*P(A)/P(B)
#A = People who have COVID
#B = People who test positive for COVID but do not have symptoms

#P_Not_B_given_A = 1 - False_Negative

#P_Not_B_given_A = False_Negative
#P_B_given_A = 1 - P_Not_B_given_A

P_B_given_A = (1-False_Positive)*People_Tested_Positive_without_symptoms # Probability that a person test +ve w/o symptoms given that the person has COVID
P_A = Fraction_Real_Positive
P_B = Fraction_with_no_Symptoms
P_A_given_B = P_B_given_A * P_A / P_B 
print(f"{P_B_given_A} * {P_A} / {P_B}")
print(f'P(A|B) = {P_A_given_B} and Fraction = {P_A_given_B/N}')

# # Q1 Part G and H
# print("\nQ1 Part G")

def part_g():
	print("\nQ1 Part G")
	
	frequency_pos_without_symptoms = 0.0
	frequency_pos_with_symptoms = 0.0
	frequency_negative = 0.0
	n_iter = 1000000
	positive_without_symptoms = 0.05
	positive_with_symptoms = 0.08
	
	for i in range(n_iter):
		random_num = random.random()
		if 0 < random_num and random_num < positive_with_symptoms:
			frequency_pos_with_symptoms = frequency_pos_with_symptoms + 1
		elif positive_with_symptoms < random_num and random_num < positive_with_symptoms + positive_without_symptoms:
			frequency_pos_without_symptoms = frequency_pos_without_symptoms + 1
		else:
			frequency_negative = frequency_negative + 1

	probability_pos_without_symptoms = frequency_pos_without_symptoms / n_iter
	probability_pos_with_symptoms = frequency_pos_with_symptoms / n_iter

	print(f"Probability of + with symp = {probability_pos_with_symptoms}")
	print(f"Probability of + without symp = {probability_pos_without_symptoms}")

part_g()








# # # Q2 Part A
def q2_partA():
	print("\nQ2 Part A")
	dataset = pd.read_csv('final2021.csv', header = None, delimiter=' ', skiprows=1)
	no_of_weeks = range(1,len(dataset)+1)
	# #no_of_positives = dataset[:][1]
	no_of_positives = dataset[1].tolist()
	plt.title("Number of people testing positive vs time")
	plt.xlabel("Week")
	plt.ylabel("Number of People Positive")
	plt.plot(no_of_weeks, no_of_positives, 'red')
	#plt.yscale("log",base=2)
	plt.show(block=False)
	plt.pause(7)
	plt.close()

	plt.title("Logarithm of the number of people testing positive vs Time")
	plt.xlabel("Week")
	plt.ylabel("Logarithm(number of people positive)")
	log_pos = np.log(no_of_positives)
	plt.plot(no_of_weeks, log_pos, 'blue')
	plt.show(block=False)
	plt.pause(7)
	plt.close()

q2_partA()



# Q2 Part B
def q2_partB():
	# Load Dataset
	dataset = pd.read_csv('final2021.csv', header = None, delimiter=' ', skiprows=1)

	# Generate an array for x which is number of rows = No. of Weeks =k
	X = np.array([x for x in range(1, len(dataset)+1)])

	# Load the people that are infected = Tested Positive
	Y_temp = dataset[1].tolist()
	#Y = np.array(Y_temp)
	P = dataset[:][1]
	ln_Y = np.log(P).replace([-float('inf')], 0)
	
	# log of the infected people to get log(Xk)
	#ln_Y = np.log(Y)
	Y = ln_Y # Updating Y to ln(Y)
	
	# plt.scatter(X, Y)
	# plt.show()

	# Starting to train the model
	m = 0			# Slope of the line initialised to 0
	c = 0			# Y-Intercept initialised to 0

	L = 0.0001      # The learning Rate = 0.0001
	n_iter = 45000  # The number of iterations to perform gradient descent

	n = float(len(X)) # Number of elements in X used to find derviative

	# Gradient Descent formula performed
	for i in range(n_iter): 
		
		Y_pred = m*X + c    # Y_Pred : Current Predicted Value of Y
		D_c = (-2/n) * sum(Y - Y_pred)        # Differential wrt c
		D_m = (-2/n) * sum(X * (Y - Y_pred))  # Differential wrt m
		
		c = c - L * D_c  # c is updated
		m = m - L * D_m  # m is updated
		
		
	print (m, c) # Value of slope and y-intercept after model is trained
	#0.11842098560963278, 0.8968232741343722
	global_m, global_c = m, c
	# Prediction: Generate new line with new values of m and c
	Y_pred = m*X + c

	# Print Real Data
	no_of_weeks = range(1,len(dataset)+1)
	no_of_positives = dataset[1].tolist()
	#log_pos = np.log(no_of_positives)
	plt.plot(no_of_weeks, ln_Y, 'blue')

	# Print Preidcted Data
	plt.plot(X, Y_pred, color='red')
	plt.title("Predicted Model (Red) vs Real Data (Blue)")
	plt.xlabel("Week (k)")
	plt.ylabel("Log of Number of infected people log(Xk)")
	plt.show()

	return m,c
m,c = q2_partB()



def grad_desc_func(X, Y):
	
	# Starting to train the model
	m = 0			# Slope of the line initialised to 0
	c = 0			# Y-Intercept initialised to 0

	L = 0.0001      # The learning Rate = 0.0001
	n_iter = 45000  # The number of iterations to perform gradient descent

	n = float(len(X)) # Number of elements in X used to find derviative

	# Gradient Descent formula performed
	for i in range(n_iter): 
		Y_pred = m*X + c    # Y_Pred : Current Predicted Value of Y
		D_c = (-2/n) * sum(Y - Y_pred)        # Differential wrt c
		D_m = (-2/n) * sum(X * (Y - Y_pred))  # Differential wrt m
		
		c = c - L * D_c  # c is updated
		m = m - L * D_m  # m is updated

	#print (m, c) # Value of slope and y-intercept after model is trained
	return m,c
		
		
	



def q2_part_e():
	# Load Dataset
	dataset = pd.read_csv('final2021.csv', header = None, delimiter=' ', skiprows=1)
	N_entries= len(dataset)
	S = 100
	
	# sample random 100 entries of dataset (with replacement)
	means_m = []
	means_c = []
	
	
	
	X = np.array([x for x in range(1, len(dataset)+1)])
	P = dataset[:][1]
	ln_Y = np.log(P).replace([-float('inf')], 0)
	Y = ln_Y # Updating Y to ln(Y)


	for i in range(S):
		print(f"\nRange = {i}")
		
		random_index = random.choices([i for i in range(len(X))], k =100) # Sample = 100 (with replacement)
		sample_x = np.array([X[i] for i in random_index]) # k for each sample
		sample_y = np.array([Y[i] for i in random_index]) # log(xk) for each sample
		X,Y = sample_x, sample_y
		
		m,c = grad_desc_func(X, Y) # Do gradient Descent and return a and log(x0)
		print(f"m= {m}")
		print(f"c= {c}")
		means_m.append(m)	# add a to the list
		means_c.append(c)	# add log(x0) to the list

	mu_m = sum(means_m) / len(means_m) # Calculate mean of a
	mu_c = sum(means_c) / len(means_c) # Calculate mean of log(x0)
	
	
	# Calculate Standard Deviation of a and log(x0)
	sigma_m = math.sqrt(sum((xi - mu_m) ** 2 for xi in means_m) / len(means_m))
	sigma_c = math.sqrt(sum((xi - mu_c) ** 2 for xi in means_c) / len(means_c))
	
	# For a, slope, Confidence Intervals ->
	cheb_lo_m, cheb_hi_m = mu_m - (sigma_m / math.sqrt(0.05 * N_entries)), \
	mu_m + (sigma_m / math.sqrt(0.05 * N_entries))
	
	clt_lo_m, clt_hi_m = mu_m - (2 * (sigma_m / math.sqrt(N_entries))), \
	mu_m + (2 * (sigma_m / math.sqrt(N_entries)))

	print("FOR m, 95%% CI - BS Chebyshev: %.6f <= X <= %.6f" % \
		(cheb_lo_m, cheb_hi_m))
	print("For m, 95%% CI - BS CLT: %.6f <= X <= %.6f" % \
		(clt_lo_m, clt_hi_m))

	# #For c, y-intercept, Confidence Intervals ->
	cheb_lo_c, cheb_hi_c = mu_c - (sigma_c / math.sqrt(0.05 * N_entries)), \
	mu_c + (sigma_c / math.sqrt(0.05 * N_entries))
	
	clt_lo_c, clt_hi_c = mu_c - (2 * (sigma_c / math.sqrt(N_entries))), \
	mu_c + (2 * (sigma_c / math.sqrt(N_entries)))

	print("FOR c, 95%% CI - BS Chebyshev: %.6f <= X <= %.6f" % \
		(cheb_lo_c, cheb_hi_c))
	print("For c, 95%% CI - BS CLT: %.6f <= X <= %.6f" % \
		(clt_lo_c, clt_hi_c))

	return (cheb_lo_m, cheb_hi_m)



def q2_part_f():
	Total = 5
	m = 0.11842098560963278
	x0 = math.exp(m) # x0
	cheb_low, cheb_upper = 0.115871, 0.120762
	#cheb_low, cheb_upper = bootstrap_new()
	random_as = np.random.uniform(cheb_low, cheb_upper, Total)
	
	f = lambda a, x0 : np.exp(a*10)*x0
	xk_value = [f(a, x0) for a in random_as]

	#print(xk_value)

	Mean = 	sum(xk_value) / len(xk_value) # Calculate mean of xk_value
	print(f"Mean = {Mean}")
	Variance = np.var(xk_value)
	print(f'\nVariance = {Variance}')

	sigma = math.sqrt(Variance)
	
	
	cheb_lo_c, cheb_hi_c = Mean - (sigma / math.sqrt(0.05 * Total)), \
	Mean + (sigma / math.sqrt(0.05 * Total))
	
	print("FOR c, 95%% CI Chebyshev: %.6f <= X <= %.6f" % \
		(cheb_lo_c, cheb_hi_c))

q2_part_f()



