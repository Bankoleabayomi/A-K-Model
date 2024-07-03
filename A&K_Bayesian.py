# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:48:55 2024

@author: banko
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from hyperopt import hp, fmin, tpe, Trials

# Load your dataset
data = pd.read_csv('C:/Users/banko/Documents/AK_trial_data.csv')

Gf = 20  # Insert the value of the gradient of velocity/shear stress applied
data['Gf'] = Gf

# Convert time from minutes to hours
data['Time'] = data['Tf'] / 60

# Create a new column for no/ni
data['ni_no'] = data['No'].max() / data['No']

# Define the model function
def A_K(Tf, Gf, Ka, Kb):
    return (Kb / Ka * Gf + (1 - Kb / Ka * Gf) * np.exp(-Ka * Gf * Tf)) ** -1

# Extract the necessary columns
Tf = data['Tf'].values
ni_no = data['ni_no'].values

# Define the objective function for Bayesian optimization
def objective(params):
    Ka, Kb = params
    ni_no_pred = A_K(Tf, Gf, Ka, Kb)
    mse = np.mean((ni_no - ni_no_pred) ** 2)
    return mse

# Define the search space
space = [
    hp.uniform('Ka', 1e-7, 1e-3),
    hp.uniform('Kb', 1e-7, 1e-3)
]

# Perform the Bayesian optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

# Extract the best parameters
Ka_best = best['Ka']
Kb_best = best['Kb']

# Print the best initial guesses
print(f"Best initial Ka: {Ka_best}")
print(f"Best initial Kb: {Kb_best}")

# Perform the curve fitting with the optimized initial guesses
popt, pcov = curve_fit(lambda Tf, Ka, Kb: A_K(Tf, Gf, Ka, Kb), Tf, ni_no, p0=[Ka_best, Kb_best])

# Extract the fitted parameters
Ka_fitted, Kb_fitted = popt

# Print the fitted parameters
print(f"Fitted Ka: {Ka_fitted}")
print(f"Fitted Kb: {Kb_fitted}")

# Generate the fitted curve
ni_no_fitted = A_K(Tf, Gf, Ka_fitted, Kb_fitted)

# Plot the results
plt.plot(data['Time'], ni_no, 'bo', label='Observed_data')
plt.plot(data['Time'], ni_no_fitted, 'r-', label='Bayessian_Fitted curve')
plt.xlabel('Time (hours)')
plt.ylabel('ni/no')
plt.legend()
plt.show()
