import math
import numpy as np

# Test Branin function, rescaled
def braninsc(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    
    term1 = x2bar - 5.1*x1bar**2/(4*math.pi**2) + 5*x1bar/math.pi - 6
    term2 = (10 - 10/(8*math.pi)) * math.cos(x1bar)
    
    y = (term1**2 + term2 - 44.81) / 51.95
    return(y)


# Simulate branin data for testing
np.random.seed(99)
n = 500
p = 2
x = np.random.uniform(size=n*p).reshape(n,p)
y = np.zeros(n)
for i in range(n):
    y[i] = braninsc(x[i,])

from openbt import OPENBT 

m = OPENBT(model = "bart", tc = 4, modelname = "branin")
fit = m.fit(x, y)