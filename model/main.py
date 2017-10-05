from __future__ import division
from random import shuffle, choice, randint
from os.path import expanduser
from numpy import log10
from scipy import stats
import numpy as np
import time
import copy
import sys
import os


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
    r, u, gr, mt, q = ps

    for j in range(sd):
        if sd == 1 and np.random.binomial(1, u) == 0: continue
        p = np.random.randint(1, 1000)
        if p not in sD:
            sD[p] = {'gr' : 10**np.random.uniform(gr, 0)}
            sD[p]['mt'] = 10**np.random.uniform(mt, 0)
            es = np.random.uniform(1, 100, 3)
            sD[p]['ef'] = es/sum(es)

        ID = time.time()
        iD[ID] = copy.copy(sD[p])
        iD[ID]['sp'] = p
        iD[ID]['x'] = 0
        iD[ID]['y'] = 0
        iD[ID]['q'] = 10**np.random.uniform(0, q)

    return [sD, iD]


def consume(iD, rD, ps):
    r, u, gr, mt, q = ps
    keys = list(iD)
    shuffle(keys)
    for k in keys:
        if len(list(rD)) == 0: return [iD, rD]
        c = choice(list(rD))
        e = iD[k]['ef'][rD[c]['t']] * iD[k]['q']

        iD[k]['q'] += min([rD[c]['v'], e])
        rD[c]['v'] -= min([rD[c]['v'], e])
        if rD[c]['v'] <= 0: del rD[c]
    return [iD, rD]


def grow(iD):
    for k, v in iD.items():
        m = v['mt']
        iD[k]['q'] -= v['gr'] * v['q']
        if v['q'] < m: del iD[k]
    return iD


def maintenance(iD):
    for k, v in iD.items():
        iD[k]['q'] -= v['mt']
        if v['q'] < v['mt']: del iD[k]
    return iD


def reproduce(sD, iD, ps, p = 0):
    for k, v in iD.items():
        if v['q'] > v['mt']*2 and np.random.binomial(1, v['gr']) == 1:
            iD[k]['q'] = v['q']*.85
            iD[k]['age']+=1
            i = time.time()
            iD[i] = copy.copy(iD[k])
            iD[i]['q']=v['q']/.85*.15
            iD[i]['age']=0
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
    r, u, gr, mt, q = ps
    for i in range(r):
        p = np.random.binomial(1, u)
        if p == 1:
            ID = time.time()
            rD[ID] = {'t' : randint(0, 2)}
            rD[ID]['v'] = 10**np.random.uniform(0, 2)
    return rD


def run_model(sim, gr, mt, q, rD = {}, sD = {}, iD = {}, ct = 0, splist2 = []):
    print '\n'
    r = 10**randint(0, 2)
    u = 10**np.random.uniform(-2, 0)
    ps = r, u, gr, mt, q

    sD, iD = immigration(sD, iD, ps, 1000)
    while ct < 300:
        iD, sD, rD, N, ct = iter_procs(iD, sD, rD, ps, ct)
        if ct > 200 and ct%10 == 0: output(iD, sD, rD, sim, ct)



for sim in range(100000):
    gr = choice([-2.5, -2.0, -1.5, -1.0])
    mt = choice([-2.0, -1.5, -1.0])
    q = choice([1, 2])
    run_model(sim, gr, mt, q)
