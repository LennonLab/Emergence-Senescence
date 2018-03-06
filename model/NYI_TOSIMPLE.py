from __future__ import division
from random import shuffle, choice, randint, seed
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

col_headers = 'sim,r,gr,mt,q,rls_min,rls_max,grcv,mtcv,rlscv,ct,rlsmean,rlsvar,total.abundance,species.richness'
OUT =open("/gpfs/home/r/z/rzmogerr/Carbonate/TOSIMPLE.csv", 'w+')
print>>OUT, col_headers
OUT.close()
    
#senesce_simple = lambda age, rls: (1-(age/(rls+0.01)))
senesce_simple = lambda age, rls: (age/age)*(rls/rls)


tradeoff_reverse_logistic = lambda rls: 2 / (2 + math.exp((0.2*rls)-8))#in the full implementation, don't enforce these parameters
#tradeoff_reverse_logistic = lambda rls: 2 / (2 + math.exp((0.2*rls)-4))
#tradeoff_reverse_logistic = lambda rls: rls/rls

g0delay = lambda rls: 1 / (1 + (rls/100))

#competitive_growth = lambda age: 

def output(iD, sD, rD, sim, ct, r):
    IndIDs, SpIDs = [], []
    for k, v in iD.items():
            IndIDs.append(k)
            SpIDs.append(v['sp'])
            
    #pp(IndIDs)
    #pp(SpIDs)

    N = len(IndIDs)
    R = len(rD.items())
    S = len(list(set(SpIDs)))
    #RLSL=[]
    #for i in IndIDs:
    #    RLSL.append(iD[i]['rls'])
    RLSL=[iD[i]['rls'] for i in IndIDs]
    rlsmean = np.mean(RLSL)
    rlsvar = np.var(RLSL)    

    if N > 0:
        #OUT = open(GenPath + 'SimData.csv', 'a')
	OUT = open("/gpfs/home/r/z/rzmogerr/Carbonate/TOSIMPLE.csv",'a+')
        outlist = [sim, r, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, ct, rlsmean, rlsvar, N, S]
        outlist = str(outlist).strip('[]')
        outlist = outlist.replace(" ", "")
        print>>OUT,outlist
        OUT.close()
    try:
        print 'sim:', '%3s' % sim, 'ct:', '%3s' % ct,'  N:', '%4s' %  N, '  S:', '%4s' %  S,  '  R:', '%4s' %  R, 'LSm:' '%1s' % rlsmean, 'LSv:' '%2s' % rlsvar
    except UnboundLocalError:
        print 'ERROR: N=0'
    return

def immigration(sD, iD, ps, sd=1):
    r, u, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a = ps

    for j in range(sd):
        if sd == 1 and np.random.binomial(1, u) == 0: continue
        p = np.random.randint(1, 1000)
        if p not in sD:
            sD[p] = {'gr' : 10**np.random.uniform(gr, 0)}
            sD[p]['mt'] = 10**np.random.uniform(mt, 0)
            sD[p]['rls'] = 50#randint(rls_min,rls_max)
            sD[p]['grcv']=10**np.random.uniform(-6.01,grcv)
            sD[p]['mtcv']=10**np.random.uniform(-6.01,mtcv)
            sD[p]['rlscv']=.15#10**np.random.uniform(-6.01,rlscv)
            sD[p]['efcv']=10**np.random.uniform(-6.01,efcv)
            es = np.random.uniform(1, 100, 3)
            sD[p]['ef'] = es/sum(es)
            sD[p]['a']=a

        ID = time.time()
        iD[ID] = copy.copy(sD[p])
        iD[ID]['sp'] = p
        iD[ID]['age']=np.random.geometric(.5)-1   
        #iD[ID]['age']=0#doesn't need to start with age==0...
        iD[ID]['x'] = 0
        iD[ID]['y'] = 0
        iD[ID]['rls']=sD[p]['rls']; iD[ID]['mt']=sD[p]['mt']; iD[ID]['ef']=sD[p]['ef'];iD[ID]['gr']=sD[p]['gr'];iD[ID]['a']=sD[p]['a']
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
                grorig=(v['gr'])/(senesce_simple(v['age'],v['rls']))
                iD[k]['gr']=v['gr']/(senesce_simple((v['age']-1),v['rls']))*(senesce_simple(v['age'],v['rls']))
                
                #modifier based on the newly incremented age value, after removing the gr reduction due to previous age
                #in full implementation the sscnc model will be chosen at random from a list of choices
                i = time.time()
                iD[i] = copy.deepcopy(iD[k])
                iD[k]['age']+=1
                #in addition to copying physiology, need to copy the rlsmax---
                #rlsmax is determined genetically so there should be a chance of mutation, here with normally distributed
                #effect sizes
                iD[i]['rls']=np.random.normal((v['rls']),sD[v['sp']]['rlscv']*v['rls'],None)
                #pp(iD[k]['age']);pp(iD[k]['rls'])
                try:
                    iD[i]['gr']=np.random.normal(grorig,(sD[v['sp']]['grcv']*grorig),None)#these should not be normal distrns, should be negv-biased
                    iD[i]['mt']=np.random.normal(v['mt'],sD[v['sp']]['mtcv']*v['mt'],None)
#is total ef allowed to != 1
                except ValueError:
                    del iD[i]; continue

                if iD[i]['gr'] > 1 or iD[i]['gr'] < 0:
                    del iD[i]; continue
                iD[i]['q']=(v['q'])/(0.5+v['a'])*(0.5-v['a'])
                iD[i]['age']=0
    return [sD, iD]

def iter_procs(iD, sD, rD, ps, ct):
    procs = range(6)
    shuffle(procs)
    for p in procs:
        if p == 0: rD = ResIn(rD, ps)
        elif p == 1: pass#sD, iD = immigration(sD, iD, ps)
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
    rD={};iD={};sD={}
    if iD=={} and sD=={} and rD=={}:
        pass
    else:
        sys.exit()
    r = choice([10,100])#10**randint(0, 2)
    u = 10**np.random.uniform(-2, 0)
    ps = r, u, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a

    sD, iD = immigration(sD, iD, ps, 1000)#this is the initial number of indivs
    while ct < 2000:#this is the number of timesteps
        if ct < 1:
            print str(rls_min) + ' ' + str(rls_max) + " " + str(r)
        iD, sD, rD, N, ct = iter_procs(iD, sD, rD, ps, ct)
        if (ct > 1400 and ct%100 == 0) or (ct == 1):           
            output(iD, sD, rD, sim, ct, r)
            

for sim in range(500):#number of different models run (had been set at 10**6)
    seed(time.time())
    gr = np.random.uniform(-2,-1)
    mt = np.random.uniform(-2,-1)
    rls_min = randint(1,10)
    rls_max = randint(rls_min,100)
    grcv = np.random.uniform(-6,-0.3)
    mtcv = np.random.uniform(-6,-0.3)
    rlscv = np.random.uniform(-6,-0.3)
    efcv = np.random.uniform(-6,-0.3)
    q = choice([1, 2])
    a=.35#a can take values [0,0.5)
    run_model(sim, gr, mt, q, rls_min, rls_max, grcv, mtcv, rlscv, efcv, a)
