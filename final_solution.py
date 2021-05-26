import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from random import uniform



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

# def func_gradient_descent(theeta_0, dfx, learning_rate, no_iterations) :
# 	list_theetas = [None]*no_iterations
# 	list_theetas[0] = theeta_0
	
# 	for index in range(1,no_iterations):
# 		# FORMULA : Theeta(i) = Theeta(i-1) - (alpha * dfx(theeta(i-1)))
# 		list_theetas[index] = list_theetas[index - 1] - (learning_rate * dfx(list_theetas[index-1],index))	

# 	return list_theetas


dataset = pd.read_csv('./final2021.csv', header=None)
#dataset = pd.read_csv('./common.csv', header=None)

# Common Dataset
#N,P,S = 71100, 10074, 2736
#N,P,S = 200, 21, 5

#Tanmay Data Set
N,P,S = 10695, 1477, 895


#Q1 Part A
print("\nQ1 Part A")
Fraction_with_no_Symptoms, Fraction_with_Symptoms, Fraction_Negative = partA_fraction(N,P,S) 
#print(f'Probabilities = {Fraction_with_no_Symptoms}, { Fraction_with_Symptoms}, {Fraction_Negative}')
print(f'Fraction of People tested +ve without symptoms = {Fraction_with_no_Symptoms}')
print(f'Fraction of People tested +ve with symptoms = {Fraction_with_Symptoms}')

#mu = (0*Fraction_Negative)+(1*Fraction_with_no_Symptoms) + (2*Fraction_with_Symptoms)
#print(f'Mean mu = {mu}')

#Q1 Part B 
print("\nQ1 Part B")
clt_lower, clt_upper = clt(Fraction_with_no_Symptoms, 71100)
print(f'CI(tested +ve but have no symptoms) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')

clt_lower, clt_upper = clt(Fraction_with_Symptoms, 71100)
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
clt_lower, clt_upper = clt(Fraction_Real_Positive, 10695)
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

# Q1 Part G
print("\nQ1 Part G")
print("TO DO")

# Q1 Part H
print("\nQ1 Part H")
print("TO DO")

# # Q2 Part A
# #### UNCOMMENT THIS TO SHOW GRAPH
# print("\nQ2 Part A")
# dataset = pd.read_csv('final2021.csv', header = None, delimiter=' ', skiprows=1)
# no_of_weeks = range(1,len(dataset)+1)
# #no_of_positives = dataset[:][1]
# no_of_positives = dataset[1].tolist()
# plt.title("Number of people testing positive vs time")
# plt.xlabel("Week")
# plt.ylabel("Number of People Positive")
# plt.plot(no_of_weeks, no_of_positives, 'red')
# #plt.yscale("log",base=2)
# plt.show(block=False)
# plt.pause(15)
# plt.close()

# plt.title("Logarithm of the number of people testing positive vs Time")
# plt.xlabel("Week")
# plt.ylabel("Logarithm(number of people positive)")
# log_pos = np.log(no_of_positives)
# plt.plot(no_of_weeks, log_pos, 'blue')
# plt.show(block=False)
# plt.pause(15)
# plt.close()




# Q2 Part B
dataset = pd.read_csv('final2021.csv', header = None, delimiter=' ', skiprows=1)


#X = range(1,len(dataset)+1)
X = np.array([x for x in range(1, len(dataset)+1)])
#no_of_positives = dataset[:][1]
Y_temp = dataset[1].tolist()
Y = np.array(Y_temp)
#Y = dataset[:][1]

#ln_Y = np.log(Y).replace([-float('math.inf')],0)
ln_Y = np.log(Y)
#ln_Y = np.log(Y).replace([-float('inf')], 0)


Y = ln_Y # Updating Y to ln(Y)

# plt.scatter(X, Y)
# plt.show()

# Building the model
m = 0			# Slope of the line initialised to 0
c = 0			# Y-Intercept initialised to 0

L = 0.0001  # The learning Rate
epochs = 45000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

#print(X[55])
# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)

# Making predictions
Y_pred = m*X + c

#plt.scatter(X, Y)
no_of_weeks = range(1,len(dataset)+1)
no_of_positives = dataset[1].tolist()
#log_pos = np.log(no_of_positives)
plt.plot(no_of_weeks, ln_Y, 'blue')

new_x = [min(X), max(X)]
new_y = [np.amin(Y_pred), np.amax(Y_pred)]
plt.plot(X, Y_pred, color='red') # predicted
plt.show()
