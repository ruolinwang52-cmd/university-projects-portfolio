#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:40:53 2021

@author: Eva, Ruolin
"""

from numpy import *
import matplotlib.pyplot as plt


# Task 1
class Interval:
    def __init__(self, left, right = None): #If no right input is given, we set right to None automatically 
            if isinstance(left, complex):
                raise TypeError("Endpoint must be a real number")
            if isinstance(right, complex):
                raise TypeError("Endpoint must be a real number")
            if right is None:      #Task 7 
                right = left
            if left>right:
                raise TypeError("Wrong bounds order")
            self.left = left
            self.right = right

#Task 2, Task 8       
    def __add__(self,other):
        if isinstance(other, Interval): #First check whether other belongs to class Interval 
            return Interval(self.left + other.left, self.right + other.right) #Adding another Interval instance
        if isinstance(other, (int, float)): #Int, float does not have left/right 
            return Interval(self.left + other, self.right + other) #Adding only a number 
    
    def __radd__(self, other):   #reverses the role of self and other,when adding a number from the left,
        if isinstance(other, (int, float)):
            return Interval(other,other) + self
        else:
            return self + other
            
    
    def __sub__(self,other):
        if isinstance(other, Interval):
            return Interval(self.left - other.right, self.right - other.left)
        if isinstance(other, (int,float)):
            return Interval(self.left - other, self.right - other)
        
    def __rsub__(self,other):
        if isinstance(other, (int, float)):
            return Interval(other,other) - self
        else:
            return self - other
    
    def __mul__(self,other):
        if isinstance(other, Interval):
            return Interval(min((self.left * other.left),(self.left * other.right),(self.right * other.left),(self.right * other.right)), max((self.left * other.left),(self.left * other.right),(self.right * other.left),(self.right * other.right)))
        if isinstance(other , (int,float)):
            return Interval(self.left * other, self.right * other)
        
    def __rmul__(self, other):
        return self * other
#Task 6    
    def __truediv__(self,other):
        if other.left * other.right == 0:
            raise ValueError("The interval in the denominator cannot conatin zero.")
        if other.left < 0 < other.right:
            raise ValueError("The resulting interval is infinitely large")
        else:
            return Interval(min((self.left / other.left),(self.left / other.right),(self.right / other.left),(self.right / other.right)), max((self.left / other.left),(self.left / other.right),(self.right / other.left),(self.right / other.right)))
#Task 3    
    def __repr__(self): # this method prints a string 
        return f"[{self.left},{self.right}]"
#Task 5    
    def __contains__(self, n):
        if n == self.left:
            return True
        elif n == self.right:
            return True
        elif (self.left < n) and (n < self.right):
            return True
        else:
            return False
            
#Task 9        
    def __pow__(self, other):
        if isinstance(other, int) and other >= 0:
            if other == 0:
                return Interval(1,1)
            if other % 2 == 1: #When n is odd
                return Interval(self.left ** other, self.right ** other)
            else:
                if self.left >= 0: #we only need to check left becuase right must be larger than left 
                    return Interval(self.left ** other, self.right ** other)
                if self.right < 0: #we only need to check right becuase left must be smaller than right 
                    return Interval(self.right ** other, self.left ** other)
                else:
                    return Interval(0, max(self.left ** other, self.right ** other))
        else:
            raise TypeError("Power should be nonegative integer")

#Task 3
print(Interval(1,2))


#Task 4
I1 = Interval(1, 4)
I2 = Interval(-2, -1)
print(I1 + I2)
print(I1 - I2)
print(I1 * I2)
print(I1 / I2)
print(1 - Interval(2,3))


#Task 5
print(I1.__contains__(4))
print(4 in I1)

#Task 6
I1 = Interval(1, 4)
I2 = Interval(-1, 0)
#print(I1 / I2)

#Task 7
print(Interval(1))


#Task 8
print(Interval(2,3) + 1)
print(1 + Interval(2,3))
print(1.0 + Interval(2,3))
print(Interval(2,3) + 1.0)
print(1 - Interval(2,3))
print(Interval(2,3) -1)
print(1.0 - Interval(2,3))
print(Interval(2,3) - 1.0)
print(Interval(2,3)*1)
print(1*Interval(2,3))
print(1.0*Interval(2,3))
print(Interval(2,3)*1.0)


#Task 9
x = Interval(-2,2)
print(x)
print(x**2)
print(x**3)


#Task 10

xl=linspace(0.,1,1000)
xu=linspace(0.,1,1000)+0.5
def p(x):
    return 3*x**3-2*x**2-5*x-1

yl = zeros(len(xl),dtype=float64)
yu = zeros(len(xl),dtype=float64)
def plot_iteration2(xscope,series1,series2, title, start=0):
    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=30, fontweight='bold')
    plt.plot(xscope[start:], series1[start:], "b-")
    plt.plot(xscope[start:], series2[start:], "g-")
    plt.xlabel("I")
    plt.ylabel("p(I)")
    plt.grid(True)
for j in range(len(xl)):
    val = p(Interval(xl[j],xu[j]))
    yl[j] = val.left
    yu[j] = val.right
plot_iteration2(xl,yl,yu,"Task 10")


