# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:53:45 2019

@author: ks2815
"""


import numpy as np
import matplotlib.pyplot as plt
import random

def trapezium(f,a,b,epsilon,v):
    inter_steps=[(b-a)*0.5*(f(a)+f(b))]
    inter_steps.append(0.5*(inter_steps[0] + (b-a)*f(0.5*(a+b))))
    epsilons=[np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1])]
    i_s=[1]
    i=2  
    while epsilon<np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]):
        epsilons.append(np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]))
        #if i>2:
            #print(np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]))
        i_s.append(i)
        h=((b-a)/(2**i))
        inter_step_2=0.5*inter_steps[-1]
        for n in range(2**(i-1)):
            inter_step_2+=h*f((a+(2*n+1)*h))
        inter_steps.append(inter_step_2)
        i+=1
    if v>0:
        print("Value of integral using trapez: %f, after %i iterations"%(inter_step_2,i))
        #print(inter_steps)
        plt.figure(0)
        plt.plot([n for n in range(len(inter_steps))],inter_steps,color="blue", linewidth=0.5,label='trapez steps')
        plt.figure(1)
        plt.plot([n for n in range(len(epsilons))],epsilons,color="green", linewidth=0.5,label='trapez epsilon')
    return inter_step_2, i, inter_steps



def a(z):
    return 2*z*np.pi
def psi_sq(z):
    return np.abs(((np.pi)**-0.25)*np.exp(0.5*(-z**2))*np.exp(1j*a(z)))**2
def func(x):
    return x**4
#i=np.arange(0,15,1)
x=np.arange(-3,3,0.01)
#plt.plot(x,psi_sq(x),color="blue", linewidth=0.5)
trapezium(psi_sq,0,2,10**-6,1)

def simp(f,a,b,epsilon):
    epsilons=[]
    inter_steps_t= trapezium(f,a,b,epsilon,0)[2]
    inter_steps=[0]*(len(inter_steps_t)-1)
    for i in range(len(inter_steps_t)-1):
        #print(i)
        inter_steps[i]=(4/3)*inter_steps_t[i+1]-(1/3)*inter_steps_t[i]
        #print(inter_steps[i])
    for j in range(len(inter_steps)):
        epsilons.append(np.abs((inter_steps[j+1]-inter_steps[j])/inter_steps[j+1]))
        if epsilon>np.abs((inter_steps[j+1]-inter_steps[j])/inter_steps[j+1]):
            integral=inter_steps[j]
            steps=j
            break
    plt.figure(0)
    plt.plot([n for n in range(len(inter_steps))],inter_steps,color="red", linewidth=0.5, label='simpson steps')
    plt.legend()
    plt.figure(1)
    plt.plot([n for n in range(len(epsilons))],epsilons,color="orange", linewidth=0.5, label='simpson epsilon')
    plt.legend()
    print("Value of integral using Simpson: %f, after %i iterations"%(integral,steps))
    return integral, i, inter_steps
    
simp(psi_sq,0,2,10**-6)

def monte_carlo(f,a,b,epsilon):
    random.uniform(0,1)
    
    
    
    
monte_carlo(psi_sq,0,2,10**-6)



    