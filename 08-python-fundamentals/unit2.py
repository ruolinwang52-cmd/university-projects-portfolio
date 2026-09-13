#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:54:31 2021

@author: wangruolin
"""

import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
from math import *

x=0.5
a=0.5
List_Y=[0]
List_X=[0]
for i in  range(200):
    List_X.append(x)
    x=sin(x)-a*x+30
    List_Y.append(x)
    K=abs(List_Y[i+1]-List_X[i+1])
    if K<1.e-8:
        break
    print(f'The  result  after {i+1} iterations  is {x} ')
    



xxx = [k for k in range(5,30)]
yyy= [sin(x)-0.5*x+30 for x in xxx]



plt.plot(xxx,yyy)


from sympy import*
from sympy.abc import*
r = limit((5-77*sin(n)+8*n**2)/(1-4*n**2),n , 00)
print(r)

Listttt =[0]



for n in range(1,9999):
    
    XN=(sin(n)**2)/n 
    
    if XN >= 1.e-9:
        Listttt.append(XN)

KK=len(Listttt)-1
print(KK)
    
