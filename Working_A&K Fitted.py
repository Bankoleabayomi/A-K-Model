# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:37:34 2024

@author: banko
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load your dataset
data = pd.read_csv('C:/Users/banko/Documents/AK_trial_data.csv')

Gf = 60  # Insert the value of the gradient of velocity/shear stress applied
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

# Initial guesses for the parameters
initial_guesses = [0.00001, 0.0000014]

# Perform the curve fitting
popt, pcov = curve_fit(lambda Tf, Ka, Kb: A_K(Tf, Gf, Ka, Kb), Tf, ni_no, p0=initial_guesses)

# Extract the fitted parameters
Ka_fitted, Kb_fitted = popt

# Print the fitted parameters
print(f"Fitted Ka: {Ka_fitted}")
print(f"Fitted Kb: {Kb_fitted}")

# Generate the fitted curve
ni_no_fitted = A_K(Tf, Gf, Ka_fitted, Kb_fitted)

# Plot the results
plt.plot(data['Time'], ni_no, 'bo', label='Data')
plt.plot(data['Time'], ni_no_fitted, 'r-', label='Fitted curve')
plt.xlabel('Time (hours)')
plt.ylabel('ni/no')
plt.legend()
plt.show()
