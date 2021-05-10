import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math




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

#dataset = pd.read_csv('./final2021.csv', header=None)
dataset = pd.read_csv('./common.csv', header=None)



#Last Row for common = 71100 10074 2736


#Q1 Part A for Common
Fraction_with_no_Symptoms, Fraction_with_Symptoms, Fraction_Negative = partA_fraction(71100, 10074, 2736) 
print(f'\n Probabilities = {Fraction_with_no_Symptoms}, { Fraction_with_Symptoms}, {Fraction_Negative}')

mu = (0*Fraction_Negative)+(1*Fraction_with_no_Symptoms) + (2*Fraction_with_no_Symptoms)
print(f'Mean mu = {mu}')
# Q1 Part B for Common
clt_lower, clt_upper = clt(0.1032067511, 71100)
print(f'\nCI(tested +ve but have no symptoms) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')

clt_lower, clt_upper = clt(0.03848101266, 71100)
print(f'\nCI(tested +ve and have symptoms) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')


#Last Row for TK = 10695 1477 895
print("\n\nXXXXXXXXX Solutions for Tanmay XXXXXXX\n\n")
#Q1 Part A for Tanmay
print("\n\nTanmay Q1 Part A")
Fraction_with_no_Symptoms, Fraction_with_Symptoms, Fraction_Negative = partA_fraction(10695, 1477, 895)
print(f'\n Means = {Fraction_with_no_Symptoms}, { Fraction_with_Symptoms}, {Fraction_Negative}')


# Q1 Part B for Tanmay
print("\n\nTanmay Q1 Part B")
clt_lower, clt_upper = clt(0.05441795231, 10695)
print(f'\nCI(tested +ve but have no symptoms) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')

clt_lower, clt_upper = clt(0.08368396447, 10695)
print(f'\nCI(tested +ve and have symptoms) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')


# Q1 Part D Common 
False_Positive = 0.01
False_Negative = 0.1

N = 71100
P = 10074
S = 2736
Fraction_with_no_Symptoms, Fraction_with_Symptoms, Fraction_Negative = partA_fraction(N, P, S) 

People_Tested_Positive_without_symptoms = Fraction_with_no_Symptoms * N
Real_Positive = ( (1-False_Positive)*People_Tested_Positive_without_symptoms) + (False_Negative*(N-P))
print(f'\n\nReal Positive = {Real_Positive} and Fraction of population which are truely positve but have no symptoms = {Real_Positive/N}')

# Q1 Part E for Tanmay
#P(A|B) = P(B|A)*P(A)/P(B)
P_Not_B_given_A = 1 - False_Negative
P_B_given_A = 1 - P_Not_B_given_A
P_A = Real_Positive*N
P_B = Fraction_with_no_Symptoms * N
P_A_given_B = P_B_given_A * P_A / P_B 
print(f'\nP(A|B) = {P_A_given_B}')

# Q1 Part D Tanmay 
False_Positive = 0.01
False_Negative = 0.1

#10695 1477 895
N = 10695
P = 1477
S = 895
Fraction_with_no_Symptoms, Fraction_with_Symptoms, Fraction_Negative = partA_fraction(N, P, S) 

People_Tested_Positive_without_symptoms = Fraction_with_no_Symptoms * N
Real_Positive = ( (1-False_Positive)*People_Tested_Positive_without_symptoms) + (False_Negative*(N-P))
Fraction_Real_Positive = Real_Positive/N
print(f'\n\nTK :Real Positive = {Real_Positive} and Fraction of population which are truely positve but have no symptoms = {Fraction_Real_Positive}')

#Q1 Part E Tanmay
print("\n\nTanmay Q1 Part E")
clt_lower, clt_upper = clt(Fraction_Real_Positive, 10695)
print(f'\nCI(Real Positive People) =  {clt_lower} ≤ CI(X) ≤{clt_upper}')



# Q1 Part F for Tanmay
#P(A|B) = P(B|A)*P(A)/P(B)
P_Not_B_given_A = 1 - False_Negative
P_B_given_A = 1 - P_Not_B_given_A
P_A = Real_Positive*N
P_B = Fraction_with_no_Symptoms * N
P_A_given_B = P_B_given_A * P_A / P_B 
print(f'\n TK: P(A|B) = {P_A_given_B}')

