import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.interpolate as interp
import os
import time


def transform_dat(m):
    N = len(m[0])
    etas = [0] + [float(i)/(N - 1) for i in range(1, N)]
    pts = [(_, _) for _ in range(N**2)]
    vals = list(pts)

    c = 0
    for i in range(N):
        for j in range(N):
            pts[c] = (etas[i], etas[j])
            vals[c] = m[i, j]
            c += 1

    return pts, vals


def open_experiment(fn, n=50):
    data = list(range(n))
    for i in range(n):
        with open(fn+"_"+str(i)+".dat") as f:
            data[i] = pickle.load(f)
    return np.mean(data, 0), np.std(data, 0)


def contour_plot(data, n=50, show=True):
    ls = np.linspace(0, 1, len(data))
    plt.contour(ls, ls, data, [0.025, 0.05, 0.1, 0.15, 0.2, 0.3], colors='k', alpha=0.2)
    plt.contourf(ls, ls, data, n, cmap=plt.cm.hot)
    if show:
        plt.show()


def std_plot(data, sd, show=True):
    ls = np.linspace(0, 1, len(data))

    plt.plot(ls, data, 'k-')
    plt.plot(ls, data + sd, 'k--', alpha=0.5)
    plt.plot(ls, data - sd, 'k--', alpha=0.5)

    if show:
        plt.show()


if __name__ == "__main__":
    start_time = time.time()
    ls = [0, 0.001, 0.025, 0.05, 0.075, 0.1]
    m = [0.27, 0.31239999999999996, 0.5448000000000001, 0.686, 0.7931999999999999, 0.8399999999999999]
    s = [0.010881176406988356, 0.01198999582985749, 0.016292329483533027, 0.010695793565696736, 0.00677052435192429, 0.009959919678390978]
    m = np.array(m)
    s = np.array(s)

    plt.plot(ls, m, 'k-')
    plt.plot(ls, m+s, 'k--')
    plt.plot(ls, m-s, 'k--')
    plt.xlabel(r"$g$")
    plt.ylabel(r"$\rho(0, g)$")
    plt.savefig("../../images/wod_in_moh.png")
    plt.clf()

    m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_1.0")

    ls = np.linspace(0, 1, len(m))

    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\rho(\eta, \eta)$")

    std_plot(np.array([m[x, x] for x in range(len(m))]), np.array([s[x, x] for x in range(len(m))]), show=False)

    plt.savefig("../../images/moh_in_wod.png")
    plt.clf()

    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.xlabel(r"$\eta_2$")
        plt.tight_layout()
        plt.ylabel(r"$\rho(%0.1f, \eta_2)$" % ls[i*7])
        std_plot(m[i*7, :], s[i*7, :], show=False)

    plt.savefig("../../images/constant_eta1_stackplot.png")
    plt.clf()

    plt.xlabel(r"$\eta_2$")
    plt.ylabel(r"$\eta_1$")
    plt.title(r"Fixation Probability as a Function of Plasticity")
    contour_plot(m, show=False)
    plt.colorbar()
    plt.savefig("../../images/contourplot.png")
    plt.clf()

    plt.xlabel(r'$\eta_2$')
    plt.ylabel('Fixation Probability')
    plt.title('Average Fixation Probability for Wild-Type de-Differentiation')
    std_plot(np.mean(m, 0), np.mean(s, 0), show=False)
    plt.savefig("../../images/avg_eta1_plot.png")
    plt.clf()

    
    plt.figure(figsize=(8.5,11))
    for i, r2 in enumerate([0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]):
        plt.subplot(4, 2, i+1)

        m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_"+str(r2))
        plt.xlabel(r'$\eta_2$')
        plt.ylabel(r'Fix. Prob.')
        plt.title(r'$r_2=%0.2f$' % r2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)

    plt.savefig("../../images/avg_eta1_r2_stackplot.png")
    plt.clf()
    plt.figure(figsize=None)

    for i, u2 in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        plt.subplot(3, 3, i + 1)

        m, s = open_experiment("../../data/vary_eta1_eta2_u2/moran_u2_" + str(u2))
        plt.xlabel(r'$\eta_2$')
        plt.ylabel(r'Fix. Prob.')
        plt.title(r'$u_2=%0.2f$' % u2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)

    plt.savefig("../../images/avg_eta1_u2_stackplot.png")
    plt.clf()

    m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_1.0")

    points, values = transform_dat(m)

    gran = 2001j

    grid_x, grid_y = np.mgrid[0:1:gran, 0:1:gran]

    gz = interp.griddata(points, values, (grid_x, grid_y), method='cubic')

    #points, values = transform_dat(s)

    #gs = interp.griddata(points, values, (grid_x, grid_y), method='cubic')

    def F(x, y, targ=gz):
        max_gran = int(gran.imag - 1)

        xiter = yiter = True

        try:
            len(x)
        except:
            xiter = False

        try:
            len(y)
        except:
            yiter = False

        if xiter and not yiter:
            y = [y for _ in range(len(x))]
        elif yiter and not xiter:
            x = [x for _ in range(len(y))]
        elif not yiter and not xiter:
            y = [y]
            x = [x]

        X, Y = np.array([int(xx * max_gran) for xx in x]), np.array([int(yy * max_gran) for yy in y])

        bool = (X >= 0) * (X <= max_gran) * (Y >= 0) * (Y <= max_gran)

        return np.array([targ[X[i], Y[i]] if bool[i] else -1 for i in range(len(X))])

    ls = np.linspace(0, 1, 36)

    M = 150
    X = np.linspace(0, 1, M + 1)

    linestyles = ["-k", "-kd", "-ko", "-k*", "-ks", "-kp", "-k^"]


    for i,alpha in enumerate((0, 0.05, 0.1, 0.2, 0.3, 0.4)):
        dat = F(X + alpha, X)

        bool = dat >= 0

        if alpha == 0:
            lab = r"$\eta_1=\eta_2$"
        else:
            lab = r"$\eta_1=\eta_2 + %0.2f$" % alpha

        plt.plot(X[bool], dat[bool], linestyles[i], markevery=10, label=lab)

    plt.xlabel(r"$\eta_2$")
    plt.ylabel("Fixation Probability")

    plt.legend()
    plt.savefig("../../images/eta_plus_const.png")
    plt.clf()


    for i,alpha in enumerate((0, 0.05, 0.1, 0.2, 0.3, 0.4)):
        dat = F(X - alpha, X)
        bool = dat >= 0

        if alpha == 0:
            lab = r"$\eta_1=\eta_2$"
        else:
            lab = r"$\eta_1=\eta_2 - %0.2f$" % alpha

        plt.plot(X[bool], dat[bool], linestyles[i], markevery=10, label=lab)

    plt.xlabel(r"$\eta_2$")
    plt.ylabel("Fixation Probability")

    plt.legend()
    plt.savefig("../../images/eta_minus_const.png")
    plt.clf()

    for i,alpha in enumerate((0, 0.2, 0.4, 0.6, 0.8)):
        dat = F(X / (1. + alpha), X)

        bool = dat >= 0

        if alpha == 0:
            lab = r"$\eta_1=\eta_2$"
        else:
            lab = r"$\eta_1=\frac{\eta_2}{%0.1f}$" % (1+alpha)

        plt.plot(X[bool], dat[bool], linestyles[i], markevery=10, label=lab)

    plt.xlabel(r"$\eta_2$")
    plt.ylabel("Fixation Probability")

    plt.legend()
    plt.savefig("../../images/eta_div_const.png")
    plt.clf()

    for i,alpha in enumerate((0, 0.2, 0.4, 0.6, 0.8)):
        dat = F(X * (1 + alpha), X)
        bool = dat >= 0

        if alpha == 0:
            lab = r"$\eta_1=\eta_2$"
        else:
            lab = r"$\eta_1=%0.1f\eta_2$" % (1 + alpha)

        plt.plot(X[bool], dat[bool], linestyles[i], markevery=10, label=lab)

    plt.xlabel(r"$\eta_2$")
    plt.ylabel("Fixation Probability")

    plt.legend()
    plt.savefig("../../images/eta_times_const.png")
    plt.clf()

    elapsed_time = time.time() - start_time

    print "Plots generated in ", int(elapsed_time / 60), "minutes and", int(elapsed_time - int(elapsed_time /60)*60), "seconds"

