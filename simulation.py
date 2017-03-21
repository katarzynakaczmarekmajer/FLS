import numpy as np
import mlpy
import matplotlib.pyplot as plt
import collections
import sklearn.svm as svm
import pylab as pl

import timeit
import __builtin__
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from pylab import *
from numpy import *
from scipy import *
import pylab as pl
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib
import numpy as np
import pymc as mc
from scipy.stats import norm
import pymc
import sys
from pymc.Matplot import plot

from scipy import stats
import pandas

import statsmodels.api as sm

from statsmodels.graphics.api import qqplot

def predictFromM001(TS_Test_Data, TS_Test_Fcst, fcstHorizon, dataSize, draws,TrueMean):
    """
    Returns array of fcstHorizon iterated forecasts due to M001
    """

    PredictionArray=[0.00]*fcstHorizon

    T = len(TS_Test_Data)
    y = TS_Test_Data[1:T-1]
    x = TS_Test_Data[0:T-2]
    n = T-2

    #print "y simulation:"
    #print y
    #raw_input("Press Enter to continue...")

    x1 = x
    x2 = norm.rvs(0, 0.01, size=n)
 #   print "x2: " + str(len(x2))
    X = np.column_stack([x1, x2])
    np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    #Pierwszy model ! 0.0-0.1-0.2->0.1, drugi: 0.3-0.4-0.5-0.6 -> 0.45, trzeci: 0.7-0.8-0.9 -> 0.8

   # beta1_ols1 = pymc.Normal('alpha1', 0.8, 0.01)

    #beta1_ols1 = pymc.Normal('alpha1', TrueMean, 0.01)
    beta1_ols1 = pymc.Uniform('alpha1', lower=-1 , upper=1)
    beta2_ols1 = pymc.Normal('error', 0, 0.01)

        #beta1_ols = pymc.Exponential('beta1',  alpha )
        #beta1_ols = pymc.Normal('beta1',  alpha, 0.01 )

    @pymc.deterministic
    def y_hat_ols1(beta1=beta1_ols1,  x1=x1, beta2=beta2_ols1):
        return beta1 * x1

    Y_ols1 = pymc.Normal(name='Y', mu=y_hat_ols1, tau=1.0, value=y, observed=True)

    ols_model1 = pymc.Model([Y_ols1, beta1_ols1, beta2_ols1])
    ols_map1 = pymc.MAP(ols_model1)
    ols_map1.fit()

    def get_coefficients(map_):
        return [{str(variable): variable.value} for variable in map_.variables if str(variable).startswith('beta')]

        #print get_coefficients(ols_map1)
    model1 = mc.Model([Y_ols1, beta1_ols1, beta2_ols1])
    mcmc1 = mc.MCMC(model1)
    mcmc1.sample(draws)
    Summary1 = mcmc1.stats()
    #print "Summary:" + str(Summary1)
    mean_alpha = np.average(mcmc1.trace('alpha1')[:]) #correspondes to the least squares estimate
    mean_error = np.average(mcmc1.trace('error')[:]) #correspondes to the least squares estimate

    if mean_error>0.5:
        print "Error przekracza 5%!"
    PredictionArray[0]=mean_alpha*TS_Test_Data[T-1]
    fcstHorizon=fcstHorizon-1
    for s in range(fcstHorizon):
        PredictionArray[s+1]=mean_alpha*PredictionArray[s]

    #print "Prediction A: " + str(prediction) + " " + str(mean_alpha) + " " + str(TS_Test_Data[T-2] ) + " " + str(mean_error)
    return PredictionArray,mean_alpha,mean_error

def predictFromM002(TS_Test_Data, TS_Test_Fcst, fcstHorizon, dataSize, draws,TrueMean):
    """
    Returns array of fcstHorizon iterated forecasts due to M001
    """

    PredictionArray=[0.00]*fcstHorizon

    T = len(TS_Test_Data)
    y = TS_Test_Data[1:T-1]
    x = TS_Test_Data[0:T-2]
    n = T-2

    #print "y simulation:"
    #print y
    #raw_input("Press Enter to continue...")

    x1 = x
    x2 = norm.rvs(0, 0.01, size=n)
 #   print "x2: " + str(len(x2))
    X = np.column_stack([x1, x2])
    np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    #Pierwszy model ! 0.0-0.1-0.2->0.1, drugi: 0.3-0.4-0.5-0.6 -> 0.45, trzeci: 0.7-0.8-0.9 -> 0.8

   # beta1_ols1 = pymc.Normal('alpha1', 0.8, 0.01)

    #beta1_ols1 = pymc.Normal('alpha1', TrueMean, 0.01)
    beta1_ols1 = pymc.Uniform('alpha1', lower=-1 , upper=-0.3)
    beta2_ols1 = pymc.Normal('error', 0, 0.01)

        #beta1_ols = pymc.Exponential('beta1',  alpha )
        #beta1_ols = pymc.Normal('beta1',  alpha, 0.01 )

    @pymc.deterministic
    def y_hat_ols1(beta1=beta1_ols1,  x1=x1, beta2=beta2_ols1):
        return beta1 * x1

    Y_ols1 = pymc.Normal(name='Y', mu=y_hat_ols1, tau=1.0, value=y, observed=True)

    ols_model1 = pymc.Model([Y_ols1, beta1_ols1, beta2_ols1])
    ols_map1 = pymc.MAP(ols_model1)
    ols_map1.fit()

    def get_coefficients(map_):
        return [{str(variable): variable.value} for variable in map_.variables if str(variable).startswith('beta')]

        #print get_coefficients(ols_map1)
    model1 = mc.Model([Y_ols1, beta1_ols1, beta2_ols1])
    mcmc1 = mc.MCMC(model1)
    mcmc1.sample(draws)
    Summary1 = mcmc1.stats()
    #print "Summary:" + str(Summary1)
    mean_alpha = np.average(mcmc1.trace('alpha1')[:]) #correspondes to the least squares estimate
    mean_error = np.average(mcmc1.trace('error')[:]) #correspondes to the least squares estimate

    if mean_error>0.5:
        print "Error przekracza 5%!"
    PredictionArray[0]=mean_alpha*TS_Test_Data[T-1]
    fcstHorizon=fcstHorizon-1
    for s in range(fcstHorizon):
        PredictionArray[s+1]=mean_alpha*PredictionArray[s]

    #print "Prediction A: " + str(prediction) + " " + str(mean_alpha) + " " + str(TS_Test_Data[T-2] ) + " " + str(mean_error)
    return PredictionArray,mean_alpha,mean_error


def predictFromM003(TS_Test_Data, TS_Test_Fcst, fcstHorizon, dataSize, draws,TrueMean):
    """
    Returns array of fcstHorizon iterated forecasts due to M001
    """

    PredictionArray=[0.00]*fcstHorizon

    T = len(TS_Test_Data)
    y = TS_Test_Data[1:T-1]
    x = TS_Test_Data[0:T-2]
    n = T-2

    #print "y simulation:"
    #print y
    #raw_input("Press Enter to continue...")

    x1 = x
    x2 = norm.rvs(0, 0.01, size=n)
 #   print "x2: " + str(len(x2))
    X = np.column_stack([x1, x2])
    np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    #Pierwszy model ! 0.0-0.1-0.2->0.1, drugi: 0.3-0.4-0.5-0.6 -> 0.45, trzeci: 0.7-0.8-0.9 -> 0.8

   # beta1_ols1 = pymc.Normal('alpha1', 0.8, 0.01)

    #beta1_ols1 = pymc.Normal('alpha1', TrueMean, 0.01)
    beta1_ols1 = pymc.Uniform('alpha1', lower=0.3 , upper=1)
    beta2_ols1 = pymc.Normal('error', 0, 0.01)

        #beta1_ols = pymc.Exponential('beta1',  alpha )
        #beta1_ols = pymc.Normal('beta1',  alpha, 0.01 )

    @pymc.deterministic
    def y_hat_ols1(beta1=beta1_ols1,  x1=x1, beta2=beta2_ols1):
        return beta1 * x1

    Y_ols1 = pymc.Normal(name='Y', mu=y_hat_ols1, tau=1.0, value=y, observed=True)

    ols_model1 = pymc.Model([Y_ols1, beta1_ols1, beta2_ols1])
    ols_map1 = pymc.MAP(ols_model1)
    ols_map1.fit()

    def get_coefficients(map_):
        return [{str(variable): variable.value} for variable in map_.variables if str(variable).startswith('beta')]

        #print get_coefficients(ols_map1)
    model1 = mc.Model([Y_ols1, beta1_ols1, beta2_ols1])
    mcmc1 = mc.MCMC(model1)
    mcmc1.sample(draws)
    Summary1 = mcmc1.stats()
    #print "Summary:" + str(Summary1)
    mean_alpha = np.average(mcmc1.trace('alpha1')[:]) #correspondes to the least squares estimate
    mean_error = np.average(mcmc1.trace('error')[:]) #correspondes to the least squares estimate

    if mean_error>0.5:
        print "Error przekracza 5%!"
    PredictionArray[0]=mean_alpha*TS_Test_Data[T-1]
    fcstHorizon=fcstHorizon-1
    for s in range(fcstHorizon):
        PredictionArray[s+1]=mean_alpha*PredictionArray[s]

    #print "Prediction A: " + str(prediction) + " " + str(mean_alpha) + " " + str(TS_Test_Data[T-2] ) + " " + str(mean_error)
    return PredictionArray,mean_alpha,mean_error

def predictFromM003yule(TS_Test_Data, TS_Test_Fcst, fcstHorizon, dataSize, draws,TrueMean):
    """
    Returns array of fcstHorizon iterated forecasts due to M001
    """
    from pylab import *
    import scipy.signal
    from spectrum import *

    T = len(TS_Test_Data)
    #print T
    #print TS_Test_Data
    #print TS_Test_Data[T-2]
    #print TS_Test_Data[T-1]
    p = Periodogram(TS_Test_Data, sampling=2)  # y is a list of list hence the y[0]
    # now, let us try to estimate the original AR parameters
    AR, P, k = aryule(TS_Test_Data, 1, norm='biased', allow_singularity=True)
    #AR,b, rho = arma_estimate(TS_Test_Data, 1, 0, 0)
    #print str(TS_Test_Data[T-1]) + " AR Yule-Walker: " + str(AR) + " P: " + str(P) + " k: " + str(k)
    mean_alpha = (-1)*AR[0] #correspondes to the least squares estimate
    prediction = mean_alpha*TS_Test_Data[T-1]

    return prediction,mean_alpha

def predictFromM004(TS_Test_Data, TS_Test_Fcst, fcstHorizon, dataSize, draws,TrueMean):
    """
    Returns array of fcstHorizon iterated forecasts due to M001
    """
    from pylab import *
    import scipy.signal
    from spectrum import *

    T = len(TS_Test_Data)
    #print T
    #print TS_Test_Data
    #print TS_Test_Data[T-2]
    #print TS_Test_Data[T-1]
    p = Periodogram(TS_Test_Data, sampling=2)  # y is a list of list hence the y[0]
    # now, let us try to estimate the original AR parameters
    AR, P, k = arburg(TS_Test_Data, 1)
    #AR, P, k = aryule(TS_Test_Data, 1)
    #print "AR Burg: " + str(AR) + " P: " + str(P) + " k: " + str(k)
    mean_alpha = (-1)*AR[0] #correspondes to the least squares estimate
    prediction = mean_alpha*TS_Test_Data[T-1]

    return prediction,mean_alpha

def predictNaive(TS_Test_Data, TS_Test_Fcst, fcstHorizon, desiredFcst):
    #(Prediction_001+Prediction_002+Prediction_003)/3
    prediction=TS_Test_Data[desiredFcst-2]
    return prediction

def predictUniform(TS_Test_Data, TS_Test_Fcst, fcstHorizon, desiredFcst,Models,i):
    prediction=0
    modele=Models[i].copy()
    #(Prediction_001+Prediction_002+Prediction_003)/3
    for j in range(0,len(Models[0])):
            prediction+=modele[j]
    prediction=prediction/len(Models[0])
    return prediction

def predictMILS(TS_Test_Data, TS_Test_Fcst, fcstHorizon, desiredFcst,pred1,pred2,pred3,Scores):

    prediction = 0.00

    prediction = float(pred1)*float(Scores[0]) + float(pred2)*float(Scores[1]) + float(pred3)*float(Scores[2])

    #estymacje=Scores.copy()
    #(Prediction_001+Prediction_002+Prediction_003)/3
    #Scores
    #for j in range(0,2):
     #       prediction+=pred3
    print "Prediction!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    print prediction
    return prediction

def predictTradAR(TS_Test_Data, TS_Test_Fcst, fcstHorizon, desiredFcst):
    prediction=0

    arma_mod20 = sm.tsa.ARMA(TS_Test_Data, (1,0)).fit()
    prediction=arma_mod20.predict()
    print "Prediction: "+ str(prediction)

    #"Paramaters of ARMA[1,0] simulation: "+ str(arma_mod20.params)

    return prediction,arma_mod20.params


