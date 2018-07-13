import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


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
    m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_1.0")

    ls = np.linspace(0, 1, len(m))

    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.xlabel(r"$\eta_2$")
        plt.tight_layout()
        plt.ylabel(r"$\rho_S(%0.1f, \eta_2)$" % ls[i*7])
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

    for i, r2 in enumerate([0.25, 0.5, 0.75, 1.0]):
        plt.subplot(2, 2, i+1)

        m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_"+str(r2))
        plt.xlabel(r'$\eta_2$')
        plt.ylabel(r'Fix. Prob.')
        plt.title(r'$r_2=%0.2f$' % r2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)

    plt.savefig("../../images/avg_eta1_r2_stackplot1.png")
    plt.clf()

    for i, r2 in enumerate([2.0, 3.0, 4.0, 5.0]):
        plt.subplot(2, 2, i+1)

        m, s = open_experiment("../../data/vary_eta1_eta2_r2/moran_r2_"+str(r2))
        plt.xlabel(r'$\eta_2$')
        plt.ylabel(r'Fix. Prob.')
        plt.title(r'$r_2=%0.2f$' % r2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)

    plt.savefig("../../images/avg_eta1_r2_stackplot2.png")
    plt.clf()

    for i, u2 in enumerate([0.05, 0.1, 0.15, 0.2]):
        plt.subplot(2, 2, i + 1)

        m, s = open_experiment("../../data/vary_eta1_eta2_u2/moran_u2_" + str(u2))
        plt.xlabel(r'$\eta_2$')
        plt.ylabel(r'Fix. Prob.')
        plt.title(r'$u_2=%0.2f$' % u2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)

    plt.savefig("../../images/avg_eta1_u2_stackplot1.png")
    plt.clf()

    for i, u2 in enumerate([.3, 0.4, 0.5, 0.6]):
        plt.subplot(2, 2, i + 1)

        m, s = open_experiment("../../data/vary_eta1_eta2_u2/moran_u2_" + str(u2))
        plt.xlabel(r'$\eta_2$')
        plt.ylabel(r'Fix. Prob.')
        plt.title(r'$u_2=%0.2f$' % u2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)

    plt.savefig("../../images/avg_eta1_u2_stackplot2.png")
    plt.clf()

    for i, u2 in enumerate([0.7, 0.8, 0.9, 0.95]):
        plt.subplot(2, 2, i + 1)

        m, s = open_experiment("../../data/vary_eta1_eta2_u2/moran_u2_" + str(u2))
        plt.xlabel(r'$\eta_2$')
        plt.ylabel(r'Fix. Prob.')
        plt.title(r'$u_2=%0.2f$' % u2)
        plt.tight_layout()
        std_plot(np.mean(m, 0), np.mean(s, 0), show=False)

    plt.savefig("../../images/avg_eta1_u2_stackplot3.png")
    plt.clf()
