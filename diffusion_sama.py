# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:41:27 2022
@author: J.N. sama
Solution of 1D diffusion equation using the crank-nicolson method and Thomas Algorithm to solve the resulting Tridiagonal matrix
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#initialisation of simulation parameters

D=0.1                                         #difusion coefficient
stept=0.005                                    #time step
stepx=0.01                                     #step hight in space
a=-np.pi/2                                     #lower wall boundary
b=np.pi/2                                      #upper wall boundary
T=30                                           #total simulation time
Dx=int((b-a)/stepx)                            #length of the space vector
Dt=int(T/stept)                                #length of time vector
sig=D*stept/(2*stepx**2)                       #sigma or lambda in the matrix
n0=2                                           #maximum equilibrium density value
#setting matrices and vectors to zero
x=np.linspace(a,b,Dx)
t=np.linspace(0,T,Dt)
N=np.zeros((Dx,Dt))                              #matrix contains the n_i^j+1 values
ni=np.zeros(Dx)                                  #vector contains the n_i^j values 
M=np.zeros((Dx,Dx))                              #constant matrix to the left
Y=np.zeros((Dx,Dx))                              #constant matrix on the right

#setting the initial values of each matrix
#B=n0*np.cos(np.pi*x/(2*b))                      #initial value n_i^0
B=n0*np.cos(x) 
N[:,0]=B
#initialising matrices
M[0,0]=1+sig
M[0,1]=-sig
M[Dx-1,Dx-2]=-sig
M[Dx-1,Dx-1]=1+sig

Y[0,0]=1-sig
Y[0,1]=sig
Y[Dx-1,Dx-2]=sig
Y[Dx-1,Dx-1]=1-sig
for i in range(1,Dx-1):
    M[i,i]=1+2*sig
    M[i,i+1]=-sig
    M[i,i-1]=-sig
    
    Y[i,i]=1-2*sig
    Y[i,i+1]=sig
    Y[i,i-1]=sig
    
#evaluating the left hand side of the matrix equation
ni=B

#Using thomas' algorithm to solve the tridigonal matrix equation MN[:,i+1]=Yni

for i in range(Dt-1):
    ni=np.dot(Y,ni)
    c=np.zeros(Dx)                             #c prime vector defined in the algorithm
    d=np.zeros(Dx)                             #d prime vector defined in the algorithm                
    
    #evaluating the c prime vector
    c[0]=M[0,1]/M[0,0]
    for j in range(1,Dx-1):
        c[j]=M[j,j+1]/(M[j,j]-M[j,j-1]*c[j-1])
    #evaluating the d prime vector
    d[0]=ni[0]/M[0,0]
    for j in range(1,Dx-1):
        d[j]=(ni[j]-M[j,j-1]*d[j-1])/(M[j,j]-M[j,j-1]*c[j-1])
    
    #solutions N^j+1
    N[Dx-1,i+1]=d[Dx-1]
    k=Dx-2
    while (k>0):
        N[k,i+1]=d[k]-c[k]*N[k+1,i+1]
        k=k-1
    ni=N[:,i+1]
    
#density plot at different time steps
plt.figure(1)
plt.plot(x[1:Dt-1],N[1:Dt-1,0],label='time step 0')
plt.plot(x[1:Dt-1],N[1:Dt-1,500],label='time step 500')
plt.plot(x[1:Dt-1],N[1:Dt-1,2000],label='time step 2000')
plt.plot(x[1:Dt-1],N[1:Dt-1,5000],label='time step 5000')
plt.xlabel('x')
plt.ylabel('density')
plt.title('snapshot of density profile at different time steps')
plt.grid()
plt.legend()
#plot of time evolution of density at x=0
plt.figure(2)
plt.plot(t,N[int(Dx/2),:])
plt.xlabel('t')
plt.ylabel('density')
plt.title('time evolution of density at x=0')
plt.grid()

#plot of colormap 
fig, ax=plt.subplots()
density=ax.pcolor(t,x,N,vmin=0.5,vmax=1.4)
fig.colorbar(density,ax=ax)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Density at each time step and position')
