"""

Run Moran simulations for cancer stem cells fixation probability. For more information see:

  Mahdipour-Shirayeh, A., Kaveh, K., Kohandel, M., and Sivaloganathan, S. (2017). 
      Phenotypic heterogeneity in modeling cancer evolution. PloS one, 12(10):e0187000.

To run the code simply execute

python moran.py r2 3

--or--

python moran.py u2 0.5

"""


import numpy as np
import pickle
import time
import os
from multiprocessing import Pool
import sys


def moh_run(RR2, R2T, UU1, UU2, N1, N2, RR1=1, R1T=1, D1=1, D2=1, D1T=1, D2T=1):
    """ 
    The actual guts of the moran simulation.
    Exact details given in the paper cited above.

    I allow for function type arguments for some parameter values because of a technical reason
    discussed in the upcoming paper (not used in this particular generating file, however)
    """
    fnx_type = type(lambda x: x + 1)

    def Nr(ns, nd):
        return r1(ND-nd)*(NS-ns)+r2(nd)*ns+r1t*(ND-nd)+r2t*nd

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
    """ The parallel work function. Runs the moran simulations over all the eta_1 and eta_2 possibilities """
    n_x = n_y = 35
    np.random.seed(int(sd)) 

    eta2s = [0] + [i / float(n_y) for i in range(1, n_y + 1)]
    eta1s = [0] + [i / float(n_x) for i in range(1, n_x + 1)]

    return [[moh_run(r2, r2t, 0.5, u2, eta1, eta2) for eta2 in eta2s] for eta1 in eta1s]


def temp_pkl(j, obj):
    """ This just writes a pickle file with the filename string I prefer """
    with open('moran_' + sys.argv[-2] + '_' + sys.argv[-1] + '_' + str(j) + '.dat', 'wb') as f:
        pickle.dump(obj, f)


def main():
    """ 
    Main function, kicks off the parallelization work.

    Runs the simulation for 10 sets of 1000 iterations.
    (I don't do all 10,000 at once, honestly, mostly just because of
    some quirks with the server I run it on not liking parallel processes
    to be running for "too long".)

    I check to make sure that I haven't hit a global total of MAX_j=50 for this
    particular parameter set (by checking the disk logs).

    This function also handles seeding the random number generators of the parallel
    processes to make sure no funny business happens there.

    """
    
    MAX_RUNS = 1000
    MAX_j = 50

    p = Pool(25)

    num_loops = 10

    temp_res = [_ for _ in range(num_save)]

    start_str = 'moran_' + sys.argv[-2] + '_' + sys.argv[-1] + '_'

    try:
        startj = max([int(x[x.rfind('_')+1:x.rfind('.')]) for x in os.listdir(os.getcwd()) if x.startswith(start_str)]) + 1
    except:
        startj = 0
    if startj >= MAX_j:
        return

    for i in range(num_loops):
        temp_res[i] = p.map(p_work, [time.time()*np.random.random() for _ in range(MAX_RUNS)])
        print "Done ", i # print progress to screen, useful because it takes a while

        
    try:
        startj = max([int(x[x.rfind('_')+1:x.rfind('.')]) for x in os.listdir(os.getcwd()) if x.startswith(start_str)]) + 1
    except:
        startj = 0
    if startj > MAX_j:
        return

    # The reason i check start_j twice, is because in practice I run this same code multiple times
    # on multiple different server instances that have access to the same /home drive. 
    # I often run into the problem where servera starts loop 2 and serverb also starts loop2 before
    # servera has finished computing the p.map step. servera finishes first, writes the loop2 file
    # then serverb finishes. If I didn't have this extra chance, which would allow serverb to write
    # the loop3 file, instead serverb would just re-write the loop2 file. Which destroys that data.
    # there's still a problem with wasted parallel units on the final loop through. But that's 
    # not really a concern. I figure i save more CPU time by just biting the bullet there than actually
    # doing a semaphore check at each iteration. 

    temp_pkl(startj, np.mean(np.mean(temp_res, 0),0))
    temp_res = [_ for _ in range(num_save)]

    return "Finished Successfully"


if __name__ == "__main__":
    whichparam = sys.argv[-2]

    if whichparam == "u2":
        u2 = float(sys.argv[-1])
        r2 = r2t = 1.1
    elif whichparam == "r2":
        u2 = 0.5
        r2 = r2t = float(sys.argv[-1])

    print main()

