#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:05:49 2021

@author: wangruolin
"""
import numpy as np 
import matplotlib.pyplot as plt

#Task 1
# define a function

def approx_ln(x, n=10):

    if n<=0:
       print("n must be larger or equal to 0") 

    if x>0:
        a = np.longdouble((x+1)/2) # Set initial values, longdouble makes the floot more precis 
        g = np.longdouble(np.sqrt(x,dtype=np.longdouble))   
        while n>0: #set iteration
            a = (a+g)/2 
            g = np.sqrt(a*g,dtype=np.longdouble) 
            n = n-1
        return np.longdouble((x-1)/a) 
    else:
        return np.nan #Function must have a return 

#Test any value here below
print(approx_ln(2,10))



    

def plot_task2(xscope,series1,series2, title):
    plt.xlabel("Iteration",fontsize=30) 
    plt.ylabel("Value",fontsize=30) 
    plt.figure(figsize=(30, 30))
    plt.suptitle(title, fontsize=35, fontweight='bold')
    plt.plot(xscope[0:], series1[0:], "r-",label="Value of ln") # items start through the rest of the array
    plt.plot(xscope[0:], series2[0:], "b-",label="Value of approximation") 
    plt.legend(loc='lower center', shadow=True, fontsize=50)
    plt.grid(True)
def plot_task2diff(xscope,series, title):
    plt.xlabel("Iteration",fontsize=30) 
    plt.ylabel("Difference",fontsize=30) 
    plt.figure(figsize=(30, 30))
    plt.suptitle(title, fontsize=35, fontweight='bold')
    plt.plot(xscope[0:], series[0:], "c-") 
    plt.grid(True)
def plot_task3(xscope,series, title):
    plt.xlabel("iteration",fontsize=30) 
    plt.ylabel("Absolute Difference",fontsize=30) 
    plt.figure(figsize=(30, 30))
    plt.suptitle(title, fontsize=35, fontweight='bold')
    plt.plot(xscope[0:], series[0:], "r-") 
    plt.grid(True)
    
    
#Task 2  #Define ploting function
Ncount=50
x_val=1.71
nr=np.zeros(Ncount,dtype=np.longdouble) #Creates a list full with zeros
ln = np.zeros(Ncount,dtype=np.longdouble)
ln_a = np.zeros(Ncount,dtype=np.longdouble)
diff = np.zeros(Ncount,dtype=np.longdouble)
y = np.log(x_val,dtype=np.longdouble) #This is a fixed value 

for i in range(1,Ncount+1):
    nr[i-1]=i #this is another way of using the append method
    ln[i-1]=y
    ln_a[i-1]=approx_ln(x_val,i)
    diff[i-1] = y-ln_a[i-1] # The difference between approximation and actual value 
 
   
 

plot_task2(nr,ln,ln_a,"Task 2") # Call this plotting defined above , bug:here it always produce a blank graph 
plot_task2diff(nr,diff,"Task 2 Difference") 






#Task 3
Ncount=50
x_val=1.41
nr=np.zeros(Ncount,dtype=np.longdouble)
ln = np.zeros(Ncount,dtype=np.longdouble)
ln_a = np.zeros(Ncount,dtype=np.longdouble)
y = np.log(x_val,dtype=np.longdouble)

for i in range(1,Ncount+1):
    nr[i-1]=i
    ln_a[i-1]=np.abs(y-approx_ln(x_val,i),dtype=np.longdouble)#Here is the absolute difference 


    

plot_task3(nr,ln_a,"Task 3")



#Task 4
def fast_approx_ln(x,n=10):
    if n<=0:
       print("n must be larger or equal to 0") 
    if x>0:
        N=n
        a=np.longdouble((x+1)/2)
        g = np.longdouble(np.sqrt(x,dtype=np.longdouble))
        d = np.zeros(shape=(n+1,n+1),dtype=np.longdouble) #Create two dimentional array with zeros, 0 to n
        d[0,0] = a
        while n>0:
            a = np.longdouble((a+g)/2)  
            g = np.sqrt(a*g,dtype=np.longdouble)
            n = n-1
            d[0,N-n] = a #loop from [0,1]to [0,N]
        dv = np.longdouble(0.25)
        for k in range(1,N+1):
            for i in range(1,N+1):
                d[k,i] = np.longdouble(d[k-1,i]-np.longdouble(dv*d[k-1,i-1]))/np.longdouble(1-dv)
            dv = np.longdouble(dv/4)
        return np.longdouble((x-1)/d[N,N])
    else:
        return np.nan



#Test any value here below
print(fast_approx_ln(2,10))



#Task 5
xvals = np.linspace(1e-8,20,500) # 500 points from 0+ to 20 in array, must use 1e-8 to avoid division by 0
yvals = np.array(list(map(lambda y: np.log(y,dtype=np.longdouble), xvals)))# use lambda functions,passes each element of given xvals
yvals2 = np.array(list(map(lambda y: fast_approx_ln(y,3), xvals)))# Two iterations and so on
yvals3 = np.array(list(map(lambda y: fast_approx_ln(y,4), xvals)))
yvals4 = np.array(list(map(lambda y: fast_approx_ln(y,5), xvals)))
yvals5 = np.array(list(map(lambda y: fast_approx_ln(y,6), xvals)))
plt.figure(figsize=(30, 30))
plt.xlabel("x") 
plt.ylabel("Error")
plt.suptitle("Task 5",fontweight='bold')
plt.yscale("log")
plt.plot(xvals, np.abs(yvals-yvals2), "b-",label="3 iterations")
plt.plot(xvals, np.abs(yvals-yvals3), "g-",label="4 iterations")
plt.plot(xvals, np.abs(yvals-yvals4), "r-",label="5 iterations")
plt.plot(xvals, np.abs(yvals-yvals5), "c-",label="6 iterations")
plt.legend(loc='upper left', shadow=True, fontsize=50) 
plt.grid(True)



