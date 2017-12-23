"""
results.py
A module to produce plots and numerical results from D0lifetime.py
I Manco 17/12/2016
"""

import D0lifetime as d
import numpy as np
from matplotlib import pyplot as plt

reload(d)

data = d.Data('all')
minimiser = d.Minimiser(data, 0.4, 0.9)

def minimisation_1D():
    out = minimiser.parabolic(1e-5)
    minimum = out.minimum()[0]
    standard_deviation = out.deviation_tau()
    out.plot_minimisation()
    return "lifetime =", minimum, "standard deviation =", standard_deviation

def minimisation_2D():
    out = minimiser.newton()
    minimum = out.minimum()[0]
    standard_deviation = out.ellipse()
    out.plot_minimisation()
    return "lifetime =", minimum, "standard deviation =", standard_deviation

def dev_vs_meas():
    """Find linear fit of standard deviation as a function of # of measurements (1D)"""
    num_meas = []
    dev1 = []
    dev2 = []
    x = []
    for i in np.arange(10, 10000, 100):
        data = d.Data(i)
        minimiser = d.Minimiser(data, 0.4, 0.9)
        out = minimiser.parabolic(1e-5)
        num_meas.append(i)
        dev1.append(0.5 * (abs(out.deviation_tau()[0]) + out.deviation_tau()[1]))     
        dev2.append(out.deviation_curv()[1])
        x.append(i)
    z1 = np.polyfit(1./np.sqrt(x), dev1, 1)
    z2 = np.polyfit(1./np.sqrt(x), dev2, 1)
    plt.close()
    plt.figure()
    plt.title("Log plot of standard deviation as a function of \n number of measurements included")
    plt.plot(np.log10(num_meas), np.log10(dev1), 'b.', markersize = 1)
    plt.plot(np.log10(range(10, 200000)), -0.5*np.log10(range(10, 200000)) + np.log10(z1[0]), 'b', label = "Change in NLL by 0.5")
    plt.plot(np.log10(num_meas), np.log10(dev2), 'g.', markersize = 1)
    plt.plot(np.log10(range(10, 200000)), -0.5*np.log10(range(10, 200000)) + np.log10(z2[0]), 'g', label = "Parabolic Curvature")
    plt.xlabel("Number of Measurements N")
    plt.ylabel("$\\sigma_$\tau$$")
    plt.legend() 
    return z1[0], z1[1], z2[0], z2[1]

def find_accuracy():
    """Return number of measurements needed to obtain accuracy of 10^-15 s."""   
    a = dev_vs_meas()
    print a
    x1 = np.full(10, 10000)
    x2 = np.full(10, 10000)
    for i in range(10):
        x1[i] = x1[i-1] - (a[0]/np.sqrt(x1[i-1]) - 1e-3)/(-0.5*a[0]/(x1[i-1])**(3./2.))
    for i in range(10):
        x2[i] = x2[i-1] - (a[2]/np.sqrt(x2[i-1]) - 1e-3)/(-0.5*a[2]/(x2[i-1])**(3./2.))
    x = np.arange(0, 1.3*np.log10(x1[-1]))
    plt.plot(x, np.full((len(x)), -3), 'r')
    plt.show()
    return x1[-1], x2[-1]