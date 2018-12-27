#!/software/.admin/bins/bin/python2.7
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


def wod_run(r1, r2, p1, p2, g1, g2, d1=1, d2=1, maxNS=10, maxND=10):
    """ 
    The actual guts of the moran simulation.
    Exact details given in the paper cited above.

    I allow for function type arguments for some parameter values because of a technical reason
    discussed in the upcoming paper (not used in this particular generating file, however)
    """

    def Nr(ns, nd):
        return r1 * (NS - ns) + r2 * ns + g1 * (ND - nd) + g2 * nd

    def Wp(ns, nd): #   W^{+}, (n_S, n_D) \mapto (n_S + 1, n_D)
        return ((NS - ns) * ns / float(NS * Nr(ns, nd))) * r2 * p2

    def Wm(ns, nd): #   W^{-}, (n_S, n_D) \mapto (n_S - 1, n_D)
        return ((NS - ns) * ns / float(NS * Nr(ns, nd))) * r1 * p1

    def Wmpp(ns, nd): # W^{-}_{++}, (n_S, n_D) \mapto (n_S - 1, n_D + 2)
        return ((NS - ns) * ns / float(NS * Nr(ns, nd))) * r2 * (1 - p2)

    def Wpmm(ns, nd): # W^{+}_{--}, (n_S, n_D) \mapto (n_S + 1, n_D - 2)
        return ((NS - ns) * ns / float(NS * Nr(ns, nd))) * r1 * (1 - p1)

    def Wpm(ns, nd): #  W^{+}_{-}, (n_S, n_D) \mapto (n_S + 1, n_D - 1)
        return ((ND - nd) * nd / float(ND * Nr(ns, nd))) * g2

    def Wmp(ns, nd): #  W^{-}_{+}, (n_S, n_D) \mapto (n_S - 1, n_D + 1)
        return ((ND - nd) * nd / float(ND * Nr(ns, nd))) * g1

    def Sum(ns, nd):
        return Wp(ns, nd) + Wm(ns, nd) + Wmpp(ns, nd) + Wpmm(ns, nd) + Wpm(ns, nd) + Wmp(ns, nd)


    NS = maxNS
    ND = maxND

    # ICs
    ns = 1
    nd = 0

    MAXT = 15000

    for t in range(MAXT):
        r = np.random.random() * Sum(ns, nd)

        if r < Wp(ns, nd):
            ns, nd = ns + 1, nd
        elif r < Wp(ns, nd) + Wm(ns, nd):
            ns, nd = ns - 1, nd
        elif r < Wp(ns, nd) + Wm(ns, nd) + Wmpp(ns, nd):
            ns, nd = ns - 1, nd + 2
        elif r < Wp(ns, nd) + Wm(ns, nd) + Wmpp(ns, nd) + Wpmm(ns, nd):
            ns, nd = ns + 1, nd - 2
        elif r < Wp(ns, nd) + Wm(ns, nd) + Wmpp(ns, nd) + Wpmm(ns, nd) + Wpm(ns, nd):
            ns, nd = ns + 1, nd - 1
        else:
            ns, nd = ns - 1, nd + 1

        if ns < 0:
            ns = 0
        elif ns > NS:
            ns = NS

        if nd < 0:
            nd = 0
        elif nd > ND:
            nd = ND

        if (ns == 0 and nd == 0) or (ns == NS and nd == ND):
            break

    if ns == NS:
        return 1
    if ns == 0:
        return 0
    else:
        return -1


def p_work(arg):
    """ The parallel work function. Runs the moran simulations over all the g_1 and g_2 possibilities """
    sd, r1, r2, p1, p2 = arg
    n_x = n_y = 35
    np.random.seed(int(sd)) 

    g2s = [0] + [i / float(n_y) for i in range(1, n_y + 1)]
    g1s = [0] + [i / float(n_x) for i in range(1, n_x + 1)]

    return [[wod_run(r1, r2, p1, p2, g1, g2) for g2 in g2s] for g1 in g1s]


def temp_pkl(j, obj):
    """ This just writes a pickle file with the filename string I prefer """
    with open(outdir + '/wod-moran_' + sys.argv[-4] + '_' + sys.argv[-3] + '_' + sys.argv[-2] + '_' + sys.argv[-1] + '_' + str(j) + '.dat', 'wb') as f:
        pickle.dump(obj, f)


def main(r1, r2, p1 , p2):
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

    temp_res = [_ for _ in range(num_loops)]

    start_str = 'wod-moran_' + sys.argv[-4] + '_' + sys.argv[-3] + '_' + sys.argv[-2] + '_' + sys.argv[-1] + '_'

    try:
        startj = max([int(x[x.rfind('_')+1:x.rfind('.')]) for x in os.listdir(outdir) if x.startswith(start_str)]) + 1
    except:
        startj = 0
    if startj >= MAX_j:
        return

    for i in range(num_loops):
        print "Starting ", i
        temp_res[i] = p.map(p_work, [(time.time()*np.random.random(), r1, r2, p1, p2) for _ in range(MAX_RUNS)])
        print "Done ", i # print progress to screen, useful because it takes a while

        
    try:
        startj = max([int(x[x.rfind('_')+1:x.rfind('.')]) for x in os.listdir(outdir) if x.startswith(start_str)]) + 1
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

    return "Finished Successfully"


if __name__ == "__main__":
    r1, r2, p1, p2 = sys.argv[-4:]

    outdir = "./"

    #print main(float(r1), float(r2), float(p1), float(p2))

    print r1, r2, p1, p2

    runs = np.array([wod_run(float(r1), float(r2), float(p1), float(p2), 0.5, 0.5) for _ in range(1000)])
    print np.mean(runs == 1)
    print np.sum(runs == -1)

