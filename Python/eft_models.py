"""
Honda EFT Models:
"""

import numpy as np
from scipy.special import kv, gamma, factorial


def f_dagger(g):
    x = 1/(32*g**2)
    K = kv(0.25,x)
    b = np.exp(x)/(2*np.sqrt(2)*g)*K
    return b


def fsg(g,ns):
    #Get the term number
    klist = np.linspace(0,ns, num = ns+1)
    sk = np.zeros((g.shape[0],ns+1))
    for k in klist:
        if k%2==0:
            c = np.sqrt(2)*gamma(k + 1/2)*(-4)**(k/2)/factorial(k/2)
            sk[:,int(k)] = (c*g**k)[:,0]
    #Get the expansion
    f = np.sum(sk,axis=1)
    return f



def flg(g,nl):
    #Get the term number
    klist = np.linspace(0,nl, num = nl+1)
    lk = np.zeros((g.shape[0],nl+1))
    for k in klist:
        #Get the coefficients
        c = gamma(k/2 + 0.25)*(-0.5)**k/(2*factorial(k))
        lk[:,int(k)] = (c*g**(-k)/np.sqrt(g))[:,0]
    
    #Get the expansion
    f = np.sum(lk,axis=1)
    return f


def get_cbar(ns):
    #Get the term number
    klist = np.linspace(0,ns, num = ns+1)
    sk = np.zeros(ns+1)
    for k in klist:
        if k%2==0:
            sk[int(k)] = np.sqrt(2)*gamma(k + 1/2)*(-4)**(k/2)/(factorial(k/2)*factorial(k))
    #Estimate cbar
    h = np.where(sk!=0)
    cbar = np.sqrt(np.mean((sk[h]**2)))
    return cbar


def get_dbar(nl):
    #Get the term number
    klist = np.linspace(0,nl, num = nl+1)
    lk = np.zeros(nl+1)
    
    for k in klist:
        #Get the coefficients
        lk[int(k)] = gamma(k/2 + 0.25)*(-0.5)**k*factorial(k)/(2*factorial(k))
    
    #Estimate dbar
    if nl < 2:
        print("Warning, nl < 2, dbar is not estimated as intended")
        dbar = 1
    else:
        #Estimate cbar using coefs of order 2 through nl
        dbar = np.sqrt(np.mean(lk[2:]**2))
    
    return dbar


def dsg(g,ns,cbar = None):
    if cbar is None:
        cbar = get_cbar(ns)
    if ns%2 == 0:
        v = (cbar**2)*(factorial(ns + 2)**2)*g**(2*ns + 4)
    else:
        v = (cbar**2)*(factorial(ns + 1)**2)*g**(2*ns + 2)
    s = np.sqrt(v)
    return s


def dlg(g,nl,dbar = None):
    if dbar is None:
        #Estimate dbar
        if nl < 2:
            print("Warning, nl < 2, dbar is not estimated as intended")
            dbar = 1
        else:
            #Estimate cbar using coefs of order 2 through nl
            #dbar = sqrt(mean(lk[-c(1,2)]^2))
            dbar = get_dbar(nl)
    #Get standard deviation
    v = (dbar**2)*(1/(factorial(nl+1)**2))*(1/g**(2*nl + 3))  
    s = np.sqrt(v)
    return s


#y_train = np.loadtxt("/home/johnyannotty/Documents/openbt/Examples/Data/honda_y_train.txt").reshape(20,1)
#x_train = np.loadtxt("/home/johnyannotty/Documents/openbt/Examples/Data/honda_x_train.txt").reshape(20,1)
