# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:17:16 2016

@author: Mikolaj Wasniewski, Krzysztof Rudas, Katarzyna Kaczmarek
"""

from numpy import *
import sklearn 
import sklearn.linear_model
import pandas
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import linear_model
import skfuzzy as fuzz
import numpy as np

a40439n = pandas.read_table("a40439n.txt")

ABPMean = a40439n.ABPMean

T = ABPMean
x = a40439n['ABPMean']

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(a40439n['ABPMean'])

a40439n.fillna(method='pad')
T = a40439n[['Elapsed time', 'ABPMean']]

        
    
        
def error(T):
    T = T.fillna(method='pad')
    model = linear_model.LinearRegression()
 
    x = T['Elapsed time'].tolist()
    y = T['ABPMean'].tolist()
    xx = array(x)
    xx.shape = (1,xx.shape[0])
    yy = array(y)
    yy.shape = (1,yy.shape[0])
    x = xx.transpose()
    y = yy.transpose()
    model.fit(x,y)
    mse = mean((model.predict(x)-y)**2)
    return(mse)


def create_segment(T, j):
    T = T.fillna(method='pad')
    model = linear_model.LinearRegression()
 
    x = T['Elapsed time'].tolist()
    y = T['ABPMean'].tolist()
    xx = array(x)
    xx.shape = (1,xx.shape[0])
    yy = array(y)
    yy.shape = (1,yy.shape[0])
    x = xx.transpose()
    y = yy.transpose()
    model.fit(x,y)
    
    poczatekR = np.array([min(x), model.coef_[0][0]*min(x) + model.intercept_])
    koniecR = np.array([max(x), model.coef_[0][0]*max(x) + model.intercept_])
    dlugoscR = ((koniecR[1]-poczatekR[1])**2 + (koniecR[0]-poczatekR[0])**2)**.5
    dynamikaR = model.coef_[0]
    zmiennoscR = np.abs(koniecR[1]-poczatekR[1])
    mseR = np.mean((model.predict(x) - y) ** 2)

    d = {
        'cecha': ['numer', 'poczatek', 'koniec', 'dlugosc', 'dynamika', 'zmiennosc', 'mse'],
        'R': [j, poczatekR, koniecR, dlugoscR[0], dynamikaR[0], zmiennoscR[0], mseR]
    }
    df = pandas.DataFrame(data=d, columns=['R'], index=d['cecha'])
    #d = np.array([j, poczatekX, poczatekY, koniecX, koniecY, dlugoscR[0], dynamikaR[0], zmiennoscR[0], mseR]
    #d.shape = 1,7
    #df = pandas.DataFrame(d, columns=['numer', 'poczatekX','poczatekY', 'koniecX', 'koniecY', 'dlugosc', 'dynamika', 'zmiennosc', 'mse'], index = j)
    return df.transpose()

def concat(seg_ts, seg_new, j):
    seg_ts = seg_ts.append(seg_new, ignore_index=True)
    return seg_ts
    
def sliding_window(T):
    T = T.fillna(method='pad')
    
    max_error = 25
    anchor = 1
    j = 1 
    while anchor < T.shape[0]:
        i = 2
        while error(T.iloc[anchor:(anchor+i),]) < max_error:
            i += 1
            if anchor + i > T.shape[0]:
                break
        if j == 1:
            seg_ts = create_segment(T.iloc[anchor:(anchor+i),], j)
        else:
            seg_new = create_segment(T.iloc[anchor:(anchor+i),], 1)
            seg_ts = concat(seg_ts, seg_new, j)
        anchor = anchor + i
        j = j + 1
        print(anchor, j)
        
    return seg_ts
    
# wykres fragmentu szeregu
df2 = sliding_window(T.iloc[1000:1300,])
plt.plot(a40439n['Elapsed time'][1000:1300], a40439n['ABPMean'][1000:1300], color = "blue")
for i in range(df2.shape[0]):
    plt.plot([df2.iloc[i,1 ][0], df2.iloc[i,2][0]],
             [df2.iloc[i,1 ][1], df2.iloc[i,2][1]], color = "red")
plt.show()

df3 = sliding_window(T)

df4 = df3.iloc[:,3:].astype('float')
df4.describe()
df4.index = np.arange(0, df4.shape[0],1)

plt.plot(a40439n['Elapsed time'], a40439n['ABPMean'], color = "blue")

###########################################
###### definicja zmiennych lingwistycznych 
duration = np.arange(df4.dlugosc.min(),df4.dlugosc.max()+1, 1)
a = df4.dlugosc.min()
b = df4.dlugosc.quantile(0.95)
duration_short = fuzz.trapmf(duration ,[a, a,a + (b-a)/4,a + (b-a)/2])
duration_medium = fuzz.trapmf(duration ,[a + (b-a)/4, a + (b-a)/3,a + (b-a)*2/3,a + (b-a)*3/4])
duration_long = fuzz.trapmf(duration ,[a + (b-a)/2 , a + (b-a)*3/4, df4.dlugosc.max(), df4.dlugosc.max()])

fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
ax0.plot(duration, duration_short, 'b', linewidth=1.5, label='duration_short')
ax0.plot(duration, duration_medium, 'g', linewidth=1.5, label='duration_medium')
ax0.plot(duration, duration_long, 'r', linewidth=1.5, label='duration_long')
ax0.set_title('Duration')
ax0.legend()
ax0.set_ylim([0, 1.05])

dynamics = np.arange(df4.dynamika.min(),df4.dynamika.max()+0.05, 0.01)
# Input Membership Functions
a = df4.dynamika.min()
b = df4.dynamika.max()
dynamics_decreasing = fuzz.trapmf(dynamics ,[a, a,a + (b-a)/4,a + (b-a)/2])
dynamics_constant = fuzz.trapmf(dynamics ,[a + (b-a)/4, a + (b-a)/3,a + (b-a)*2/3,a + (b-a)*3/4])
dynamics_increasing = fuzz.trapmf(dynamics ,[a + (b-a)/2 , a + (b-a)*3/4, b, b])

fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
ax0.plot(dynamics, dynamics_decreasing, 'b', linewidth=1.5, label='dynamics_decreasing')
ax0.plot(dynamics, dynamics_constant, 'g', linewidth=1.5, label='dynamics_constant')
ax0.plot(dynamics, dynamics_increasing, 'r', linewidth=1.5, label='dynamics_increasing')
ax0.set_title('Dynamics')
ax0.legend(loc = 'lower left')
ax0.set_ylim([0, 1.05])


variability = np.arange(df4.zmiennosc.min(),df4.zmiennosc.max()+0.1, 0.1)

a = df4.zmiennosc.min()
b = df4.zmiennosc.quantile(0.95)
variability_low = fuzz.trapmf(variability, [a, a,a + (b-a)/4,a + (b-a)/2])
variability_medium = fuzz.trapmf(variability, 
                                 [a + (b-a)/4, a + (b-a)/3,a + (b-a)*2/3,a + (b-a)*3/4])
variability_high = fuzz.trapmf(variability, 
                               [a + (b-a)/2 , a + (b-a)*3/4, df4.zmiennosc.max(), df4.zmiennosc.max()])

fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
ax0.plot(variability, variability_low, 'b', linewidth=1.5, label='variability_low')
ax0.plot(variability, variability_medium, 'g', linewidth=1.5, label='variability_medium')
ax0.plot(variability, variability_high, 'r', linewidth=1.5, label='variability_high')
ax0.set_title('Variability')
ax0.legend(loc = 'lower left')
ax0.set_ylim([0, 1.05])


mse = np.arange(df4.mse.min(),df4.mse.max()+0.1, 0.1)

a = 25 # bo taki jest błąd w naszym algorytmie segmentacji
b = df4.mse.quantile(0.95)
mse_low = fuzz.trapmf(mse, [df4.mse.min(), df4.mse.min(),a + (b-a)/4,a + (b-a)/2])
mse_medium = fuzz.trapmf(mse, 
                                 [a + (b-a)/4, a + (b-a)/3,a + (b-a)*2/3,a + (b-a)*3/4])
mse_high = fuzz.trapmf(mse, 
                               [a + (b-a)/2 , a + (b-a)*3/4, df4.mse.max(), df4.mse.max()])

fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
ax0.plot(mse, mse_low, 'b', linewidth=1.5, label='mse_low')
ax0.plot(mse, mse_medium, 'g', linewidth=1.5, label='mse_medium')
ax0.plot(mse, mse_high, 'r', linewidth=1.5, label='mse_high')
ax0.set_title('mse')
ax0.legend(loc = 'lower left')
ax0.set_ylim([0, 1.05])


def kwantyfikator(x):
    czesc = np.arange(0, 1.01, 0.01)
    
    wiekszosc = fuzz.trapmf(czesc, [0.45, 0.6, 1, 1])
    mniejszosc = fuzz.trapmf(czesc, [0, 0, 0.4, 0.55])
    prawie_wszystkie = fuzz.trapmf(czesc, [0.7, 0.8, 1, 1])
    
    czesc_wiekszeosc = fuzz.interp_membership(czesc,wiekszosc, x) # Depends from Step 1
    czesc_mniejszosc = fuzz.interp_membership(czesc,mniejszosc, x)
    czesc_prawie_wszystkie = fuzz.interp_membership(czesc, prawie_wszystkie, x)

    return dict(wiekszosc = czesc_wiekszeosc, mniejszosc = czesc_mniejszosc, 
                prawie_wszystkie = czesc_prawie_wszystkie)

czesc = np.arange(0, 1.01, 0.01)

wiekszosc = fuzz.trapmf(czesc, [0.45, 0.6, 1, 1])
mniejszosc = fuzz.trapmf(czesc, [0, 0, 0.4, 0.55])
prawie_wszystkie = fuzz.trapmf(czesc, [0.7, 0.8, 1, 1])
fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
ax0.plot(czesc, wiekszosc, 'b', linewidth=1.5, label='wiekszosc')
ax0.plot(czesc, mniejszosc, 'g', linewidth=1.5, label='mniejszosc')
ax0.plot(czesc, prawie_wszystkie, 'r', linewidth=1.5, label='prawie_wszystkie')
ax0.set_title('kwantyfikator')
ax0.legend(loc = 'lower left')
ax0.set_ylim([0, 1.05])


#################
# Funkcje do fuzzyfikacji odpowiednich segmentow szeregu  
def mse_category(mse_in = 0):
    mse_cat_low = fuzz.interp_membership(mse, mse_low, mse_in) 
    mse_cat_medium = fuzz.interp_membership(mse, mse_medium, mse_in)
    mse_cat_high = fuzz.interp_membership(mse, mse_high, mse_in)
    
    return pandas.DataFrame([[mse_cat_low, mse_cat_medium, mse_cat_high]], 
                            columns = ["mse_low", "mse_medium", "mse_high"])


def mse_table(df):
    n = df.shape[0]
    for i in range(n):
        if i == 0:
            d = mse_category(df.mse[i])
        else:
            d = d.append(mse_category(df.mse[i]),
                         ignore_index=True)
    return d

def variability_category(variability_in = 0):
    variability_cat_low = fuzz.interp_membership(variability, variability_low, variability_in) # Depends from Step 1
    variability_cat_medium = fuzz.interp_membership(variability, variability_medium, variability_in)
    variability_cat_high = fuzz.interp_membership(variability, variability_high, variability_in)
    
    return pandas.DataFrame([[variability_cat_low, variability_cat_medium, variability_cat_high]], 
                            columns = ["variability_low", "variability_medium", "variability_high"])


def variability_table(df):
    n = df.shape[0]
    for i in range(n):
        if i == 0:
            d = variability_category(df.zmiennosc[i])
        else:
            d = d.append(variability_category(df.zmiennosc[i]),
                         ignore_index=True)
    return d


def duration_category(duration_in = 150):
    duration_cat_short = fuzz.interp_membership(duration,duration_short, duration_in) # Depends from Step 1
    duration_cat_medium = fuzz.interp_membership(duration,duration_medium, duration_in)
    duration_cat_long = fuzz.interp_membership(duration, duration_long, duration_in)
    
    return pandas.DataFrame([[duration_cat_short, duration_cat_medium, duration_cat_long]], 
                            columns = ["duration_short", "duration_medium", "duration_long"])



def duration_table(df):
    n = df.shape[0]
    for i in range(n):
        if i == 0:
            d = duration_category(df.dlugosc[i])
        else:
            d = d.append(duration_category(df.dlugosc[i]),
                         ignore_index=True)
    return d




def dynamics_category(dynamics_in = 0):
    dynamics_cat_decreasing = fuzz.interp_membership(dynamics,dynamics_decreasing, dynamics_in) # Depends from Step 1
    dynamics_cat_constant = fuzz.interp_membership(dynamics,dynamics_constant, dynamics_in)
    dynamics_cat_increasing = fuzz.interp_membership(dynamics, dynamics_increasing, dynamics_in)
    
    return pandas.DataFrame([[dynamics_cat_decreasing, dynamics_cat_constant, dynamics_cat_increasing]], 
                            columns = ["dynamics_decreasing", "dynamics_constant", "dynamics_increasing"])


def dynamics_table(df):
    n = df.shape[0]
    for i in range(n):
        if i == 0:
            d = dynamics_category(df.dynamika[i])
        else:
            d = d.append(dynamics_category(df.dynamika[i]),
                         ignore_index=True)
    return d

def characteristic_table(df):
    d1 = duration_table(df)
    d2 = dynamics_table(df)
    d3 = variability_table(df)
    d4 = mse_table(df)
    return pandas.concat([d1, d2, d3, d4], axis = 1)

d = characteristic_table(df4)



def Degree_of_truth_ext(d, Q = "mniejszosc", P = "duration_long", R = "dynamics_decreasing"):    
    """
    Stopień prawdy dla zlozonych podsumowan lingwistycznych
    """    
    p = np.fmin(d[P], d[R])
    r = d[R]
    t = np.sum(p)/np.sum(r)
    return kwantyfikator(t)[Q]
    
Degree_of_truth_ext(d = d)  

    



    
def Degree_of_truth(d = d, Q = "mniejszosc", P = "duration_long", P2 = ""):
    """
    Stopień prawdy dla prostych podsumowan lingwistycznych
    """    
    if P2 == "":    
        p = np.mean(d[P])
    else:
        p = np.mean(np.fmin(d[P], d[P2]))
    return kwantyfikator(p)[Q]

Degree_of_truth()

Degree_of_truth(d = d, Q = "wiekszosc", P = "dynamics_decreasing")
Degree_of_truth(d = d, Q = "wiekszosc", P = "duration_short")
Degree_of_truth(d = d, Q = "wiekszosc", P = "duration_short", P2 = "dynamics_decreasing")
Degree_of_truth_ext(d = d, Q = "wiekszosc", P = "duration_long", R = "dynamics_increasing")



    
def all_protoform(d):
    """
    Funkcja wyznaczajoca stopnie prawdy dla wszystkich 
    podumowań lingwistycznych (prostych i zlozonych)    
    """
    pp = ["duration_short", "duration_medium", "duration_long"]
    rr = ["dynamics_decreasing", "dynamics_constant", "dynamics_increasing"]
    qq = ["variability_low", "variability_medium", "variability_high"]
    zz = ["mse_low", "mse_medium", "mse_high"]
    protoform = np.empty(120, dtype = "object")
    DoT = np.zeros(120)
    k = 0
    for i in range(len(pp)):
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = qq[i])
        protoform[k] = "Among all trends, most are " + qq[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = pp[i])
        protoform[k] = "Among all trends, most are " + pp[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = rr[i])
        protoform[k] =  "Among all trends, most are " + rr[i]
        k += 1
        DoT[k] = Degree_of_truth(d = d, Q = "wiekszosc", P = zz[i])
        protoform[k] =  "Among all trends, most are " + zz[i]
        k += 1
    for i in range(len(pp)):
        for j in range(len(rr)):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = qq[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " treds, most are " + qq[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = rr[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " treds, most are " + rr[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = zz[j], R = pp[i])
            protoform[k] = "Among all "+ pp[i] + " treds, most are " + zz[j]
            k += 1
    for i in range(len(pp)):
        for j in range(len(rr)):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = qq[j], R = rr[i])
            protoform[k] = "Among all "+ rr[i] + " treds, most are " + qq[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = pp[j], R = rr[i])
            protoform[k] = "Among all "+ rr[i] + " treds, most are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = zz[j], R = rr[i])
            protoform[k] = "Among all "+ rr[i] + " treds, most are " + zz[j]
            k += 1

    for i in range(len(pp)):
        for j in range(len(rr)):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = rr[j], R = qq[i])
            protoform[k] = "Among all "+ qq[i] + " treds, most are " + rr[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = pp[j], R = qq[i])
            protoform[k] = "Among all "+ qq[i] + " treds, most are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = zz[j], R = qq[i])
            protoform[k] = "Among all "+ qq[i] + " treds, most are " + zz[j]
            k += 1
            
    for i in range(len(pp)):
        for j in range(len(rr)):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = rr[j], R = zz[i])
            protoform[k] = "Among all "+ zz[i] + " treds, most are " + rr[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = pp[j], R = zz[i])
            protoform[k] = "Among all "+ zz[i] + " treds, most are " + pp[j]
            k += 1
        for j in range(3):
            DoT[k] = Degree_of_truth_ext(d = d, Q = "wiekszosc", P = qq[j], R = zz[i])
            protoform[k] = "Among all "+ zz[i] + " treds, most are " + qq[j]
            k += 1
   
    dd = {"protoform":protoform,
         "DoT":DoT}
    dd = pandas.DataFrame(dd)   
    return dd[['protoform', "DoT"]]
         
pandas.set_option('max_colwidth',70)
df_protoform = all_protoform(d)

# 40 najbardzien prawdziwych podsumowan lingwistycznych 
df_protoform.sort('DoT', ascending = False).head(n = 40)
