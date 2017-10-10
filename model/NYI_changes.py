from __future__ import division
from random import shuffle, choice, randint
from os.path import expanduser
from numpy import log10
from scipy import stats
import numpy as np
import time
import math
import copy
import sys
import os
from pprint import pprint as pp


mydir = expanduser("~/")
sys.path.append(mydir + "GitHub/Emergence-Senescence/model")
GenPath = mydir + "GitHub/Emergence-Senescence/results/simulated_data/"

col_headers = 'sim,gr,mt,q,ct,total.abundance,species.richness,simpson.e,N.max,logmod.skew'
OUT = open(GenPath + 'SimData.csv', 'w+')
print>>OUT, col_headers
OUT.close()


def GetRAD(vector):
    RAD = []
    unique = list(set(vector))
    for val in unique: RAD.append(vector.count(val))
    return RAD, unique


def e_simpson(sad):
    sad = filter(lambda a: a != 0, sad)
    D = 0.0
    N = sum(sad)
    S = len(sad)
    for x in sad: D += (x*x) / (N*N)
    E = round((1.0/D)/S, 4)
    return E
    
senesce_simple = lambda age, rls: 1-(age/rls)

tradeoff_reverse_logistic = lambda rls: 2 / (2 + math.exp((0.2*rls)-12))#in the full implementation, don't enforce these parameters

g0delay = lambda rls: 1 / (1 + (rls/100))

#competitive_growth = lambda age: 

def output(iD, sD, rD, sim, ct):
    IndIDs, SpIDs = [], []
    for k, v in iD.items():
            IndIDs.append(k)
            SpIDs.append(v['sp'])

    N = len(IndIDs)
    R = len(rD.items())
    S = len(list(set(SpIDs)))

    if N > 0:
        RAD, splist = GetRAD(SpIDs)
        ES = e_simpson(RAD)
        Nm = max(RAD)

        skew = stats.skew(RAD)
        lms = log10(abs(float(skew)) + 1)
        if skew < 0: lms = lms * -1

        OUT = open(GenPath + 'SimData.csv', 'a')
        outlist = [sim, gr, mt, q, ct, N, S, ES, Nm, lms]
        outlist = str(outlist).strip('[]')
        outlist = outlist.replace(" ", "")
        print>>OUT, outlist
        OUT.close()

    print 'sim:', '%3s' % sim, 'ct:', '%3s' % ct,'  N:', '%4s' %  N, '  S:', '%4s' %  S,  '  R:', '%4s' %  R
    return

def immigration(sD, iD, ps, sd=1):
    r, u, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a = ps

    for j in range(sd):
        if sd == 1 and np.random.binomial(1, u) == 0: continue
        p = np.random.randint(1, 1000)
        if p not in sD:
            sD[p] = {'gr-unconstrained' : 10**np.random.uniform(gr, 0)}
            sD[p]['mt'] = 10**np.random.uniform(mt, 0)
            sD[p]['rls'] = 25#randint(rls_min,rls_max)
            temp=sD[p]['gr-unconstrained']; sD[p]['gr']=np.random.uniform(temp*tradeoff_reverse_logistic(sD[p]['rls']),temp)
            sD[p]['elto']=(sD[p]['gr']/sD[p]['gr-unconstrained'])
            sD[p]['grcv']=10**np.random.uniform(grcv,-0.3)
            sD[p]['mtcv']=10**np.random.uniform(mtcv,-0.3)
            sD[p]['rlscv']=10**np.random.uniform(rlscv,-0.3)
            sD[p]['efcv']=10**np.random.uniform(efcv,-0.3)
            es = np.random.uniform(1, 100, 3)
            sD[p]['ef'] = es/sum(es)
            sD[p]['a']=a

        ID = time.time()
        iD[ID] = copy.copy(sD[p])
        iD[ID]['sp'] = p
        iD[ID]['age']=np.random.geometric(.5)-1   
        #iD[ID]['age']=0#doesn't need to start with age==0...
        #print iD[ID]['sp']
        #pp(sD)
        iD[ID]['x'] = 0
        iD[ID]['y'] = 0
        iD[ID]['rls']=sD[p]['rls']; iD[ID]['mt']=sD[p]['mt']; iD[ID]['ef']=sD[p]['ef']
        temp2=sD[p]['gr']
        if iD[ID]['age']<2:
            iD[ID]['gr']=temp2*(g0delay(iD[ID]['rls'])**2)
            iD[ID]['g1gr']=g0delay(iD[ID]['rls'])
        else:
            iD[ID]['gr']=sD[p]['gr']
            iD[ID]['g1gr']=1
        iD[ID]['a']=sD[p]['a']
        iD[ID]['elto']=(iD[ID]['gr'])/(sD[p]['gr'])
        iD[ID]['q'] = 10**np.random.uniform(0, q)
    return [sD, iD]

def consume(iD, rD, ps):
    r, u, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a = ps
    keys = list(iD)
    shuffle(keys)
    for k in keys:
        if len(list(rD)) == 0: return [iD, rD]
        c = choice(list(rD))
        e = iD[k]['ef'][rD[c]['t']] * iD[k]['q']#why does this dep on the indiv's q? 
        #pp(iD[k]['ef'][rD[c]['t']])
        #pp(e)
        #To account for the Frenk et al. 2017, one idea that you had was to make the indiv a generalist by taking a max of
        #iD[k]['ef'][rD[c]['t']] and another number (e.g., (1/3))
        #but it would be better to do some distrn that has age as a param, so that it is generalizable and can be randomized.
        iD[k]['q'] += min([rD[c]['v'], e])
        rD[c]['v'] -= min([rD[c]['v'], e])
        if rD[c]['v'] <= 0: del rD[c]
    return [iD, rD]


def grow(iD):
    for k, v in iD.items():
        m = v['mt']
        iD[k]['q'] -= v['gr'] * (v['q'])
        if v['age']==0 and v['q'] < m/(0.5+v['a'])*(0.5-v['a']):#daughters are born in G0 phase,we know that
        #theyre smaller in G0. We don't want to kill them all because of it, though
            del iD[k]
        elif v['q'] < m:
            del iD[k]
    return iD


def maintenance(iD):#mt is less for juveniles
    for k, v in iD.items():
        if v['age']==0:
            iD[k]['q'] -= v['mt']/(0.5+v['a'])*(0.5-v['a'])
            if v['q'] < v['mt']/(0.5+v['a'])*(0.5-v['a']): del iD[k]
        else:
            iD[k]['q'] -= v['mt']
            if v['q'] < v['mt']: del iD[k]
    return iD

def reproduce(sD, iD, ps, p = 0):
    for k, v in iD.items():
        if v['gr'] > 1 or v['gr'] < 0:
            del iD[k]
        elif v['q'] > v['mt']/(0.5+v['a']) and np.random.binomial(1, v['gr']) == 1:
            if v['age'] >= v['rls'] or v['mt']<0:
                del iD[k]
            else:
                iD[k]['q'] = v['q']*(0.5+v['a'])
                iD[k]['age']+=1
                if iD[k]['age']==2:
                    iD[k]['gr']=v['gr']/v['g1gr']
                iD[k]['gr']=v['gr']/(senesce_simple((v['age']-1),v['rls']))*(senesce_simple(v['age'],v['rls']))
                
                #modifier based on the newly incremented age value, after removing the gr reduction due to previous age
                #in full implementation the sscnc model will be chosen at random from a list of choices
                i = time.time()
                iD[i] = copy.deepcopy(iD[k])
                #in addition to copying physiology, need to copy the rlsmax---
                #rlsmax is determined genetically so there should be a chance of mutation, here with normally distributed
                #effect sizes
                iD[i]['rls']=np.random.normal(v['rls'],sD[v['sp']]['rlscv']*v['rls'],None)
                try:
                    iD[i]['gr']=np.random.normal(v['gr'],sD[v['sp']]['grcv']*v['gr'],None)#these should not be normal distrns, should be negv-biased
                    iD[i]['mt']=np.random.normal(v['mt'],sD[v['sp']]['mtcv']*v['mt'],None)
#is total ef allowed to != 1
                except ValueError:
                    del iD[i]; continue

                if iD[i]['gr'] > 1 or iD[i]['gr'] < 0:
                    del iD[i]; continue
                '''for index,e in enumerate(iD[i]['ef']):
                    iD[i]['ef'][index]=np.random.normal(v['ef'][index],sD[v['sp']]['efcv']*v['ef'][index],None)      
                pp(sum(iD[i]['ef']))'''

                #instead of messing with the efs, you had the idea of giving another entry in the iD
                #which specfies the function of competitiveness gain for aging cells...                
                
                #temp3=iD[i]['gr']
                #iD[i]['gr']=np.random.uniform(temp3*tradeoff_reverse_logistic(iD[ID]['rls']),temp3)
                #iD[i]['elto']=(iD[i]['gr'])/(temp3)
                temp4=iD[i]['gr']
                del iD[i]; continue
                iD[i]['gr']=temp4*(g0delay(v['rls'])**2)
                iD[i]['g1gr']=g0delay(v['rls'])
                iD[i]['q']=(v['q'])/(0.5+v['a'])*(0.5-v['a'])
                iD[i]['age']=0
                #if iD[i]['q']>1:
                #    iD[i]['q']=1
                #need to add an 'age' initiator to the run_model function, I think
                #temp=np.random.uniform(np.random.normal(v['gr'],v['grcv']*v['gr'],None)
                #iD[i]['gr']=np.random.uniform(temp*tradeoff_reverse_logistic(iD[i]['rls']),temp)
                '''by determining tradeoff at the species level instead of at indiv level,
                lineages are able to evolve reduced tradeoffs
                if I want to make the tradeoffs stickier, I re-enforce them at the indiv level.
                I also am struggling to add early-life tradeoffs using the species-level thing :/'''
    return [sD, iD]

def iter_procs(iD, sD, rD, ps, ct):
    procs = range(6)
    shuffle(procs)
    for p in procs:
        if p == 0: rD = ResIn(rD, ps)
        elif p == 1: sD, iD = immigration(sD, iD, ps)
        elif p == 2: iD, rD = consume(iD, rD, ps)
        elif p == 3: iD = grow(iD)
        elif p == 4: iD = maintenance(iD)
        elif p == 5: sD, iD = reproduce(sD, iD, ps)
    N = len(list(iD))
    return [iD, sD, rD, N, ct+1]

def ResIn(rD, ps):
    r, u, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a = ps
    for i in range(r):
        p = np.random.binomial(1, u)
        if p == 1:
            ID = time.time()
            rD[ID] = {'t' : randint(0, 2)}
            rD[ID]['v'] = 10**np.random.uniform(0, 2)
    return rD

def run_model(sim, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a=0, rD = {}, sD = {}, iD = {}, ct = 0, splist2 = []):
    print '\n'
    r = 10**randint(0, 2)
    u = 10**np.random.uniform(-2, 0)
    ps = r, u, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a

    sD, iD = immigration(sD, iD, ps, 1000)#this is the initial number of indivs
    while ct < 300:#this is the number of timesteps
        iD, sD, rD, N, ct = iter_procs(iD, sD, rD, ps, ct)
        if ct > 200 and ct%10 == 0: output(iD, sD, rD, sim, ct)

for sim in range(10):#number of different models run (had been set at 10**6)
    gr = np.random.uniform(-2,-1)
    mt = np.random.uniform(-2,-1)
    rls_min = randint(1,10)
    rls_max = randint(rls_min,100)
    grcv = np.random.uniform(-6,-5)
    mtcv = np.random.uniform(-6,-5)
    rlscv = np.random.uniform(-6,-5)
    efcv = np.random.uniform(-6,-5)
    q = choice([1, 2])
    a=.35#a can take values [0,0.5)
    run_model(sim, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a)