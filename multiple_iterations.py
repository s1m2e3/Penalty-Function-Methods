from utils import generate_penalty_term
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def function(x):
    return x**2-2

x = np.linspace(-5,5,100)
y = function(x)
max_bound = 1
plt.plot(x,y)
num = [y]    
df = pd.DataFrame([x,y]).T
df.columns = ["x","iter_0"]
for i in range(5):
    iter_name = "iter_"+str(i+1)
    upper_bound_penalty = np.exp(y-max_bound)
    lower_bound_penalty = np.exp(-y-max_bound)
    y = y - upper_bound_penalty + lower_bound_penalty
    df[iter_name] = y
df.to_csv("multiple_iterations.csv")