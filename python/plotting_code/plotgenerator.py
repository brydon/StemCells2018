import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.interpolate as interp
import os
import sys
import time


def transform_dat(mat):
    n = len(mat[0])
    etas = [0.] + [float(_)/(n - 1) for _ in range(1, n)]
    pts = [(float(_), float(_)) for _ in range(n**2)]
    vals = list(pts)

    c = 0
    for row in range(n):
        for col in range(n):
            pts[c] = (etas[row], etas[col])
            vals[c] = mat[row, col]
            c += 1

    return np.array(pts), np.array(vals)


def open_experiment(fn, n=50):
    data = list(range(n))
    for _ in range(n):
        with open(fn+"_"+str(_)+".dat") as _f:
            data[_] = pickle.load(_f)
    return np.mean(data, 0), np.std(data, 0)


def contour_plot(data, n=50, show=True):
    _ls = np.linspace(0, 1, len(data))
    plt.contour(_ls, _ls, data, [0.025, 0.05, 0.1, 0.15, 0.2, 0.3], colors='k', alpha=0.2)
    plt.contourf(_ls, _ls, data, n, cmap=plt.cm.hot)
    if show:
        plt.show()


def std_plot(data, sd, show=True):
    _ls = np.linspace(0, 1, len(data))

    plt.plot(_ls, data, 'k-', linewidth=2)
    plt.plot(_ls, data + sd, 'k--', alpha=0.5, linewidth=2)
    plt.plot(_ls, data - sd, 'k--', alpha=0.5, linewidth=2)

    if show:
        plt.show()


if __name__ == "__main__":
    start_time = time.time()
    ls = [0, 0.001, 0.025, 0.05, 0.075, 0.1]
    m = [0.27, 0.31239999999999996, 0.5448000000000001, 0.686, 0.7931999999999999, 0.8399999999999999]
    s = [0.010881176406988356, 0.01198999582985749, 0.016292329483533027,
         0.010695793565696736, 0.00677052435192429, 0.009959919678390978]
    m = np.array(m)
    s = np.array(s)

    ''' Figure 4b '''

    f = plt.figure()
    plt.plot(ls, m, 'k-', linewidth=2)
    plt.plot(ls, m+s, 'k--', linewidth=2, alpha=0.5)
    plt.plot(ls, m-s, 'k--', linewidth=2, alpha=0.5)
    plt.xlabel(r"$g$", {'size': '15'})
    plt.ylabel(r"$\rho(0, g)$", {'size': '15'})
    f.savefig("../../images/wod_in_moh.pdf", bbox_inches="tight")
    plt.clf()

    m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_1.0")

    ''' Figure 4a '''
    f = plt.figure()

    ls = np.linspace(0, 1, len(m))

    plt.xlabel(r"$\eta$", {'size': '15'})
    plt.ylabel(r"$\rho(\eta, \eta)$", {'size': '15'})

    std_plot(np.array([m[x, x] for x in range(len(m))]), np.array([s[x, x] for x in range(len(m))]), show=False)

    f.savefig("../../images/moh_in_wod.pdf", bbox_inches="tight")
    plt.clf()

    ''' Figure 2a '''

    f = plt.figure()

    plt.xlabel(r"Plasticity, $\eta$", {'size': '15'})
    plt.ylabel(r"Fixation Probability, $\rho(0, \eta)$", {'size': '15'})

    std_plot(m[0, :], s[0, :], show=False)

    f.savefig("../../images/moh_combine.pdf", bbox_inches="tight")
    plt.clf()

    ''' Figure 6 '''

    f = plt.figure()

    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.xlabel(r"$\eta_2$")
        plt.tight_layout()
        plt.ylabel(r"$\rho(%0.1f, \eta_2)$" % ls[i*7])
        std_plot(m[i*7, :], s[i*7, :], show=False)
        plt.xticks([0, 1])
        locs, labs = plt.yticks()
        plt.yticks([0, locs[-1]])

    f.savefig("../../images/constant_eta1_stackplot.pdf", bbox_inches="tight")
    plt.clf()

    ''' Figure 5 '''

    f = plt.figure()
    plt.xlabel(r"Mutant-Type Plasticity, $\eta_2$")
    plt.ylabel(r"Wild-Type Plasticity, $\eta_1$")
    plt.title(r"Fixation Probability as a Function of Plasticity")
    contour_plot(m, show=False)
    plt.subplots_adjust(top=0.93)
    plt.colorbar()
    f.savefig("../../images/contourplot.pdf", bbox_inches="tight")
    plt.clf()

    ''' Figure 8 '''

    f = plt.figure()
    plt.xlabel(r'$\eta_2$')
    plt.ylabel('Fixation Probability')
    plt.title('Average Fixation Probability for Wild-Type de-Differentiation')
    std_plot(np.mean(m, 0), np.mean(s, 0), show=False)
    plt.subplots_adjust(top=0.93)
    f.savefig("../../images/avg_eta1_plot.pdf", bbox_inches="tight")
    plt.clf()

    ''' Figure 9 '''

    f = plt.figure(figsize=(8.5, 11))
    for i, r2 in enumerate([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]):
        plt.subplot(4, 2, i+1)

        m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_"+str(r2))
        plt.xlabel(r'$\eta_2$', {'size': '15'})
        plt.ylabel(r'Fix. Prob.', {'size': '15'})
        plt.title(r'$r_2=%0.2f$' % r2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)

    f.savefig("../../images/avg_eta1_r2_stackplot.pdf", bbox_inches="tight")
    plt.clf()
    f = plt.figure(figsize=None)

    ''' Figure 10 '''

    for i, u2 in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        plt.subplot(3, 3, i + 1)

        m, s = open_experiment("../../data/vary_eta1_eta2_u2/moran_u2_" + str(u2))
        plt.xlabel(r'$\eta_2$')
        plt.ylabel(r'Fix. Prob.')
        plt.title(r'$u_2=%0.2f$' % u2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)
        plt.xticks([0, 1])
        locs, labs = plt.yticks()
        plt.yticks([0, locs[-1]])

    plt.savefig("../../images/avg_eta1_u2_stackplot.pdf", bbox_inches="tight")
    plt.clf()

    ''' Figure 7 '''
    m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_1.0")

    points, values = transform_dat(m)

    gran = 2001j

    grid_x, grid_y = np.mgrid[0:1:gran, 0:1:gran]

    gz = interp.griddata(points, values, (grid_x, grid_y), method='cubic')

    def interp_f(in_x, in_y, targ=gz):
        max_gran = int(gran.imag - 1)

        xiter = yiter = True

        try:
            len(in_x)
        except TypeError:
            xiter = False

        try:
            len(in_y)
        except TypeError:
            yiter = False

        _y, _x = in_y, in_x

        if xiter and not yiter:
            _y = [in_y for _ in range(len(in_x))]
        elif yiter and not xiter:
            _x = [in_x for _ in range(len(in_y))]
        elif not yiter and not xiter:
            _y = [in_y]
            _x = [in_x]

        _X, _Y = np.array([int(xx * max_gran) for xx in _x]), np.array([int(yy * max_gran) for yy in _y])

        _boole = (_X >= 0) * (_X <= max_gran) * (_Y >= 0) * (_Y <= max_gran)

        return np.array([targ[_X[_i], _Y[_i]] if _boole[_i] else -1 for _i in range(len(_X))])

    ls = np.linspace(0, 1, 36)

    M = 150
    X = np.linspace(0, 1, M + 1)

    linestyles = ["-k", "-kd", "-ko", "-k*", "-ks", "-kp", "-k^"]

    f = plt.figure()

    for i, alpha in enumerate((0, 0.05, 0.1, 0.2, 0.3, 0.4)):
        dat = interp_f(X + alpha, X)

        boole = dat >= 0

        if alpha == 0:
            lab = r"$\eta_1=\eta_2$"
        else:
            lab = r"$\eta_1=\eta_2 + %0.2f$" % alpha

        plt.plot(X[boole], dat[boole], linestyles[i], markevery=10, label=lab)

    plt.legend(prop={'size': '13'})

    plt.xlabel(r"$\eta_2$", {'size': '15'})
    plt.ylabel("Fixation Probability", {'size': '15'})

    plt.legend()
    f.savefig("../../images/eta_plus_const.pdf", bbox_inches="tight")
    plt.clf()

    f = plt.figure()
    for i, alpha in enumerate((0, 0.05, 0.1, 0.2, 0.3, 0.4)):
        dat = interp_f(X - alpha, X)
        boole = dat >= 0

        if alpha == 0:
            lab = r"$\eta_1=\eta_2$"
        else:
            lab = r"$\eta_1=\eta_2 - %0.2f$" % alpha

        plt.plot(X[boole], dat[boole], linestyles[i], markevery=10, label=lab)

    plt.xlabel(r"$\eta_2$", {'size': '15'})
    plt.ylabel("Fixation Probability", {'size': '15'})

    plt.legend()
    plt.savefig("../../images/eta_minus_const.pdf", bbox_inches="tight")
    plt.clf()

    f = plt.figure()

    for i, alpha in enumerate((0, 0.2, 0.4, 0.6, 0.8)):
        dat = interp_f(X / (1. + alpha), X)

        boole = dat >= 0

        if alpha == 0:
            lab = r"$\eta_1=\eta_2$"
        else:
            lab = r"$\eta_1=\eta_2\cdot\left(%0.1f\right)^{-1}$" % (1+alpha)

        plt.plot(X[boole], dat[boole], linestyles[i], markevery=10, label=lab)

    plt.xlabel(r"$\eta_2$", {'size': '15'})
    plt.ylabel("Fixation Probability", {'size': '15'})

    plt.legend()
    f.savefig("../../images/eta_div_const.pdf", bbox_inches="tight")
    plt.clf()

    f = plt.figure()

    for i, alpha in enumerate((0, 0.2, 0.4, 0.6, 0.8)):
        dat = interp_f(X * (1 + alpha), X)
        boole = dat >= 0

        if alpha == 0:
            lab = r"$\eta_1=\eta_2$"
        else:
            lab = r"$\eta_1=%0.1f\eta_2$" % (1 + alpha)

        plt.plot(X[boole], dat[boole], linestyles[i], markevery=10, label=lab)

    plt.xlabel(r"$\eta_2$", {'size': '15'})
    plt.ylabel("Fixation Probability", {'size': '15'})

    plt.legend()
    f.savefig("../../images/eta_times_const.pdf", bbox_inches="tight")
    plt.clf()

    elapsed_time = time.time() - start_time

    print "Plots generated in", \
        int(elapsed_time / 60), "minutes and", int(elapsed_time - int(elapsed_time / 60)*60), "seconds"
