from __future__ import division
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import log10
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve
import math


def GetRAD(vector):
    RAD = []
    unique = list(set(vector))

    for val in unique:
        RAD.append(vector.count(val)) # the abundance of each Sp_

    return RAD, unique # the rad and the specieslist



############### RARITY #########################################################
def Rlogskew(sad):

    sad = filter(lambda a: a != 0, sad)

    x = sum(1 for n in sad if n < 0)
    if x >= 1:
        return float('NaN')


    S = len(sad)

    if S <= 2.0:
        return float('NaN')

    if max(sad) == min(sad):
        return float('NaN')

    sad = np.log10(sad)
    mu = np.mean(sad)

    num = 0
    denom = 0
    for ni in sad:
        num += ((ni - mu)**3.0)/S
        denom += ((ni - mu)**2.0)/S

    t1 = num/(denom**(3.0/2.0))
    t2 = (S/(S - 2.0)) * np.sqrt((S - 1.0)/S)

    return round(t1 * t2, 4)


############### LOGNORMAL VARIABLES ############################################

def Preston(sad):

    sad = filter(lambda a: a != 0, sad)

    x = sum(1 for n in sad if n < 0)
    if x >= 1:
        return float('NaN')

    N = sum(sad)

    if N <= 0:
        return float('NaN')

    Nmax = max(sad)

    left = (2 * N)/(np.sqrt(np.pi) * Nmax)

    func = lambda a : left - (math.erf(np.log(2)/a) / a)

    guess = 0.1 # alpha is often ~0.2, but appears to be lower for larger N
    a = fsolve(func, guess)

    expS = (np.sqrt(np.pi) / a) * np.exp( (np.log(2)/(2*a))**2 )

    return a[0], expS[0]


############### DOMINANCE ######################################################
def Berger_Parker(sad):

    sad = filter(lambda a: a != 0, sad)

    if sum(sad) <= 0:
        return float('NaN')

    x = sum(1 for n in sad if n < 0)
    if x >= 1:
        return float('NaN')

    return max(sad)/sum(sad)




############ DIVERSITY #########################################################
def Shannons_H(sad):

    sad = filter(lambda a: a != 0, sad)

    if sum(sad) <= 0:
        return float('NaN')

    x = sum(1 for n in sad if n < 0)
    if x >= 1:
        return float('NaN')

    sad = filter(lambda a: a != 0, sad)

    if sum(sad) == 0:
        return float('NaN')

    H = 0
    for i in sad:
        p = i/sum(sad)
        H += p*np.log(p)
    return round(H*-1.0, 6)


def simpsons_dom(sad):

    sad = filter(lambda a: a != 0, sad)

    if sum(sad) <= 0:
        return float('NaN')

    x = sum(1 for n in sad if n < 0)
    if x >= 1:
        return float('NaN')

    sad = filter(lambda a: a != 0, sad)

    if sum(sad) == 0:
        return float('NaN')


    D = 0.0
    N = sum(sad)

    for x in sad:
        D += x*x
    D = 1 - (D/(N*N))

    return D


######### EVENNESS #############################################################


def e_shannon(sad):

    sad = filter(lambda a: a != 0, sad)

    if len(sad) <= 1:
        return float('NaN')

    if sum(sad) <= 0:
        return float('NaN')

    x = sum(1 for n in sad if n < 0)
    if x >= 1:
        return float('NaN')

    sad = filter(lambda a: a != 0, sad)

    if sum(sad) == 0:
        return float('NaN')


    H = Shannons_H(sad)
    S = len(sad)
    return round(H/np.log(S), 6)




def e_simpson(sad): # based on 1/D, not 1 - D
    sad = filter(lambda a: a != 0, sad)

    D = 0.0
    N = sum(sad)
    S = len(sad)

    for x in sad:
        D += (x*x) / (N*N)

    E = round((1.0/D)/S, 4)

    if E < 0.0 or E > 1.0:
        print 'Simpsons Evenness =',E
    return E



def e_var(sad):
    sad = filter(lambda a: a != 0, sad)

    P = np.log(sad)
    S = len(sad)
    mean = np.mean(P)
    X = 0
    for x in P:
        X += (x - mean)**2/S
    evar = 1.0 - 2/np.pi*np.arctan(X)

    if evar < 0.0 or evar > 1.0:
        print 'Evar =',evar
    return evar



def get_modal(_list):

    """ Finds the mode from a kernel density function across a sample """
    exp_mode = 0.0
    density = gaussian_kde(_list)
    n = len(_list)
    xs = np.linspace(min(_list),max(_list),n)
    density.covariance_factor = lambda : .001
    density._compute_covariance()
    D = [xs,density(xs)]
    d = 0
    maxd = 0.0
    while d < len(D[1]):
        if D[1][d] > maxd:
            maxd = D[1][d]
            exp_mode = D[0][d]
        d += 1
    return exp_mode



def get_kdens_choose_kernel(xlist,kernel):
    """ Finds the kernel density function across a sample of sads """
    density = gaussian_kde(xlist)
    n = len(xlist)
    xs = np.linspace(min(xlist),max(xlist),n)
    #xs = np.linspace(0.0,1.0,n)
    density.covariance_factor = lambda : kernel
    density._compute_covariance()
    D = [xs,density(xs)]
    return D



def get_kdens(xlist):
    """ Finds the kernel density function across a sample of sads """
    density = gaussian_kde(xlist)
    #xs = np.linspace(min(xlist),max(xlist),n)
    xs = np.linspace(0.0,1.0,len(xlist))
    density.covariance_factor = lambda : 0.5
    density._compute_covariance()
    D = [xs,density(xs)]
    return D


def jaccard(seq1, seq2):

    """ Obtained from: https://github.com/doukremt/distance/blob/master/distance/_simpledists.py
        on Sept 8 2015

    Compute the Jaccard distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.
    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """

    set1, set2 = set(seq1), set(seq2)
    return 1 - len(set1 & set2) / float(len(set1 | set2))


def sorensen(seq1, seq2):

    if len(seq1) == 0 and len(seq2) == 0:
      return 0
    elif len(seq1) == 0 or len(seq2) == 0:
      return 1.0
    """ Obtained from: https://github.com/doukremt/distance/blob/master/distance/_simpledists.py
        on Sept 8 2015

    Compute the Sorensen distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.
    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """

    set1, set2 = set(seq1), set(seq2)
    return 1 - (2 * len(set1 & set2) / float(len(set1) + len(set2)))


def WhittakersTurnover(site1, site2):

  """ citation: """
  if len(site1) == 0 or len(site2) == 0:
      return float('NaN')

  set1 = set(site1)
  set2 = set(site2)

  gamma = set1.intersection(set2)         # Gamma species pool
  s = len(gamma)                                   # Gamma richness
  abar = np.mean([len(set1), len(set2)])   # Mean sample richness
  bw   = ((len(set1) - s) + (len(set2) - s))/abar

  return bw
