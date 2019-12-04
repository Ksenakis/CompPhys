# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:53:45 2019

@author: ks2815
"""


import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(846456)

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
        plt.plot([n for n in range(len(epsilons))],epsilons,color="blue", linewidth=0.5,label='trapez epsilon')
    return inter_step_2, i, inter_steps





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
    plt.figure(1)
    plt.plot([n for n in range(len(epsilons))],epsilons,color="red", linewidth=0.5, label='simpson epsilon')
    print("Value of integral using Simpson: %f, after %i iterations"%(integral,steps))
    return integral, i, inter_steps


"""
def monte_carlo_uni(f,a,b,epsilon):
    step=1
    n=step
    inter_steps=[(b-a)*(1/n)*sum(f(x) for x in np.array(np.random.uniform(a,b,step)))]
    inter_steps.append((inter_steps[-1]*n+(b-a)*sum(f(x) for x in np.array(np.random.uniform(a,b,step))))*(1/(n+step)))
    n+=step
    epsilons=[np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1])]
    while epsilon/100<np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]):
        n+=step
        epsilons.append(np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]))
        inter_steps.append((inter_steps[-1]*n+sum(f(x) for x in np.array(np.random.uniform(a,b,step)))*(b-a))*(1/(n+step)))
    print("Value of integral using monte carlo uniform: %f, after %i iterations with err %f"%(inter_steps[-1],n, epsilons[-1]))
    plt.figure(0)
    plt.plot([n for n in range(len(inter_steps))],inter_steps,color="purple", linewidth=0.5,label='monte carlo steps')
    plt.legend()
    plt.figure(1)
    plt.plot([n for n in range(len(epsilons))],epsilons,color="purple", linewidth=0.5,label='monte carlo epsilon')
    plt.legend()
    return inter_steps[-1], n



def monte_carlo_lin(f,a,b,epsilon):
    step=1
    n=step
    def g(x):
        return -0.48*x+0.98
    def inv_g(x):
        return (2*0.98-np.sqrt(4*0.98**2-8*0.48*x))/(2*0.48)

    new_rando=np.random.uniform(0,1,step)
    random_actual=[inv_g(new_rando[0])]
    inter_steps=[(1./n)*sum(f(inv_g(x))/g(inv_g(x)) for x in np.array(new_rando))]
    new_rando=np.random.uniform(0,1,step)
    random_actual.append(inv_g(new_rando[0]))
    inter_steps.append((inter_steps[-1]*n+sum(f(inv_g(x))/g(inv_g(x)) for x in np.array(new_rando)))*(1./(n+step)))
    n+=step
    epsilons=[np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1])]
    while epsilon<np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]):
        
        epsilons.append(np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]))
        new_rando=np.random.uniform(0,1,step)
        random_actual.append(inv_g(new_rando[0]))
        inter_steps.append((inter_steps[-1]*n+sum(f(inv_g(x))/g(inv_g(x)) for x in np.array(new_rando)))*(1./(n+step)))
        n+=step
    print("Value of integral using monte carlo lin: %f, after %i iterations with err %f"%(inter_steps[-1],n, epsilons[-1]))
    plt.figure(0)
    plt.plot([n for n in range(len(inter_steps))],inter_steps,color="green", linewidth=0.5,label='monte carlo lin steps')
    plt.legend()
    plt.figure(1)
    plt.plot([n for n in range(len(epsilons))],epsilons,color="green", linewidth=0.5,label='monte carlo lin epsilon')
    plt.legend()
    
    plt.figure(2)
    x_test = np.arange(0,2,0.01)
    y_test = g(x_test)
    y_psi = f(x_test)
    plt.plot(x_test, y_test, label='linear func')
    plt.plot(x_test, y_psi, label='psi_sq')
    #accepted = [(inv_g(x)) for x in np.random.uniform(0,1,10000)]
    plt.hist(random_actual, density=1, label='rando vals')
    plt.legend()
    #print(random_actual)
    return inter_steps[-1], n
    

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

simp(psi_sq,0,2,10**-6)

monte_carlo_uni(psi_sq,0,2,10**-8)
monte_carlo_lin(psi_sq,0,2,10**-8)
#plt.figure(2)
#x_test = np.arange(0,2,0.01)
#y_test = g(x_test)
#plt.plot(x_test, y_test)
#accepted = [inv_g(x) for x in np.random.uniform(0,1,100)]
#plt.hist(accepted, density=1)
"""

"""
def monte_carlo_uni(f,a,b,epsilon):
    step=1
    n=step
    inter_steps=[(b-a)*(1/n)*sum(f(x) for x in np.array(np.random.uniform(a,b,step)))]
    inter_steps.append((inter_steps[-1]*n+(b-a)*sum(f(x) for x in np.array(np.random.uniform(a,b,step))))*(1/(n+step)))
    n+=step
    epsilons=[np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1])]
    while epsilon/100<np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]):
        n+=step
        epsilons.append(np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]))
        inter_steps.append((inter_steps[-1]*n+sum(f(x) for x in np.array(np.random.uniform(a,b,step)))*(b-a))*(1/(n+step)))
    print("Value of integral using monte carlo uniform: %f, after %i iterations with err %f"%(inter_steps[-1],n, epsilons[-1]))
    plt.figure(0)
    plt.plot([n for n in range(len(inter_steps))],inter_steps,color="purple", linewidth=0.5,label='monte carlo steps')
    plt.legend()
    plt.figure(1)
    plt.plot([n for n in range(len(epsilons))],epsilons,color="purple", linewidth=0.5,label='monte carlo epsilon')
    plt.legend()
    return inter_steps[-1], n
"""


def monte_carlo_lin(f,a,b,epsilon,g,inv_g):
    step=1
    n=step

    new_rando=np.random.uniform(0,1,step)
    random_actual=[inv_g(new_rando[0])]
    inter_steps=[(1./n)*sum(f(inv_g(x))/g(inv_g(x)) for x in np.array(new_rando))]
    new_rando=np.random.uniform(0,1,step)
    random_actual.append(inv_g(new_rando[0]))
    inter_steps.append((inter_steps[-1]*n+sum(f(inv_g(x))/g(inv_g(x)) for x in np.array(new_rando)))*(1./(n+step)))
    n+=step
    epsilons=[np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1])]
    while epsilon<np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]):
        
        epsilons.append(np.abs((inter_steps[-1]-inter_steps[-2])/inter_steps[-1]))
        new_rando=np.random.uniform(0,1,step)
        random_actual.append(inv_g(new_rando[0]))
        inter_steps.append((inter_steps[-1]*n+sum(f(inv_g(x))/g(inv_g(x)) for x in np.array(new_rando)))*(1./(n+step)))
        n+=step
    print("Value of integral using monte carlo lin: %f, after %i iterations with err %f"%(inter_steps[-1],n, epsilons[-1]))
    plt.figure(0)
    plt.plot([n for n in range(len(inter_steps))],inter_steps,color="green", linewidth=0.5,label='monte carlo lin steps')
    plt.legend()
    plt.figure(1)
    plt.plot([n for n in range(len(epsilons))],epsilons,color="green", linewidth=0.5,label='monte carlo lin epsilon')
    plt.legend()
    
    plt.figure(2)
    x_test = np.arange(0,2,0.01)
    y_test = [g(x) for x in x_test]
    y_f = f(x_test)
    plt.plot(x_test, y_test, label='func')
    plt.plot(x_test, y_f, label='f')
    plt.hist(random_actual, density=2, label='rando vals')
    plt.legend()
    return inter_steps[-1], n
    

def a(z):
    return 2*z*np.pi
def psi_sq(z):
    return np.abs(((np.pi)**-0.25)*np.exp(0.5*(-z**2))*np.exp(1j*a(z)))**2
def func(x):
    return x**4
def g(x):
    return -0.48*x+0.98
def inv_g(x):
    return (2*0.98-np.sqrt(4*0.98**2-8*0.48*x))/(2*0.48)
def uni_g(x):
    return 0.5
def uni_inv_g(x):
    return 2*x
def test(x):
    return x**2



x=np.arange(-3,3,0.01)
#trapezium(psi_sq,0,2,10**-6,1)    
#simp(psi_sq,0,2,10**-6)

#monte_carlo_uni(psi_sq,0,2,10**-6)
#monte_carlo_lin(psi_sq,0,2,10**-8,g,inv_g)
#monte_carlo_lin(psi_sq,0,2,10**-8,uni_g,uni_inv_g)

monte_carlo_lin(test,0,2,10**-8,g,inv_g)
monte_carlo_lin(test,0,2,10**-8,uni_g,uni_inv_g)






    