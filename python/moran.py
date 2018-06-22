#!/software/.admin/bins/bin/python2.7

import numpy as np
import pickle
import time
import os
from multiprocessing import Pool
import sys


def moh_run(RR2, R2T, UU1, UU2, N1, N2, RR1=1, R1T=1, D1=1, D2=1, D1T=1, D2T=1):
    fnx_type = type(lambda x: x + 1)

    def Nr(ns, nd):
        return r1(ND-nd)*(NS-ns)+r2(nd)*ns+r1t*(ND-nd)+r2t*nd

    # r2 = r2(D2) = r2(nd)
    # r1 = r1(D1) = r1(ND-nd)
    # u1 = u1(D1) = u1(ND-nd)
    # u2 = u2(D2) = u2(nd)

    def Wsp(ns, nd):
        return (r2(nd) * (1 - u2(nd)) * ns + r2t * n2 * nd) * (NS - ns) / float(NS * Nr(ns, nd))

    def Wsm(ns, nd):
        return (r1(ND - nd) * (1 - u1(ND - nd)) * (NS - ns) + r1t * n1 * (ND - nd)) * ns / float(NS * Nr(ns, nd))

    def Wdp(ns, nd):
        return (r2t * (1 - n2) * nd + r2(nd) * u2(nd) * ns) * (ND - nd) / float(ND * Nr(ns, nd))

    def Wdm(ns, nd):
        return (r1t * (1 - n1) * (ND - nd) + r1(ND - nd) * u1(ND - nd) * (NS - ns)) * nd / float(ND * Nr(ns, nd))

    def Sum(ns, nd):
        return Wsp(ns, nd) + Wsm(ns, nd) + Wdp(ns, nd) + Wdm(ns, nd)

    if type(RR2) == fnx_type:
        R1, R2, U1, U2 = RR1, RR2, UU1, UU2
    else:
        R1, R2, U1, U2 = lambda d: RR1, lambda d: RR2, lambda d: UU1, lambda d: UU2

    NS = ND = 10
    r1 = R1
    r1t = R1T
    n1 = N1
    n2 = N2
    d1 = D1
    d2 = D2
    d1t = D1T
    d2t = D2T

    # The parm point
    r2 = R2
    r2t = R2T
    u1 = U1
    u2 = U2

    # ICs
    ns = 1
    nd = 0

    MAXT = 15000

    for t in range(MAXT):
        r = np.random.random() * Sum(ns, nd)

        if r < Wsp(ns, nd):
            ns += 1
        elif r < Wsp(ns, nd) + Wsm(ns, nd):
            ns -= 1
        elif r < Wsp(ns, nd) + Wsm(ns, nd) + Wdp(ns, nd):
            nd += 1
        else:
            nd -= 1

        if ns == 0 or ns == NS:
            break

    return ns == NS


def p_work(sd):
    n_x = n_y = 35
    np.random.seed(int(sd*1))

    eta2s = [0] + [i / float(n_y) for i in range(1, n_y + 1)]
    eta1s = [0] + [i / float(n_x) for i in range(1, n_x + 1)]

    return [[moh_run(u2, u2, 0.5, 0.5, eta1, eta2) for eta2 in eta2s] for eta1 in eta1s]


def temp_pkl(j, obj):
    with open('moran_u2_'+str(u2)+'_'+str(j)+'.dat', 'wb') as f:
        pickle.dump(obj, f)


def p_main():
    MAX_RUNS = 1000

    p = Pool(25)

    num_loops = 10
    num_save = 10

    temp_res = [_ for _ in range(num_save)]

    try:
        startj = max([int(x[x.rfind('_')+1:x.rfind('.')]) for x in os.listdir(os.getcwd()) if x.startswith('moran_u2_' + str(u2)+ "_")]) + 1
    except:
        startj = 0
    if startj > 49:
        return

    for i in range(num_loops):
        k = i % num_save
        temp_res[k] = p.map(p_work, [time.time()*np.random.random() for _ in range(MAX_RUNS)])
        print "Done ", i
        if k == num_save - 1:
            try:
                startj = max([int(x[x.rfind('_')+1:x.rfind('.')]) for x in os.listdir(os.getcwd()) if x.startswith('moran_u2_' + str(u2)+ "_")]) + 1
            except:
                startj = 0
            if startj > 49:
                return
            temp_pkl(startj + i/num_save, np.mean(np.mean(temp_res, 0),0))
            temp_res = [_ for _ in range(num_save)]
            print "Pickled"

    return "Finished Successfully"


if __name__ == "__main__":
    u2 = sys.argv[-1]

    try:
        u2 = float(u2)
    except ValueError:
        u2 = 0.5

    print u2

    print p_main()

