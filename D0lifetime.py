"""
D0lifetime.py
A module with the log-likelihood fit for extracting the D0 lifetime
I Manco 17/12/2016

Classes: - Data
         - Minimiser
         - Parameters
"""

import numpy as np
from scipy import special as sp
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Data:
    def __init__(self, num_meas):
        """Initialise set of n pairs of (time and sigma) measurements up to 10,000 pairs."""
        self.num_meas = num_meas
        input_file = np.loadtxt('lifetime.txt')
        self.input_file = input_file
        if num_meas == "all":
            self.time = input_file[:, 0]
            self.sigma = input_file[:, 1] 
        else:
            self.time = input_file[:num_meas, 0]
            self.sigma = input_file[:num_meas, 1] 
    
    def sorted_data(self):
        """Return data sorted in ascending order for time."""
        sorted_data = self.input_file[self.input_file[:, 0].argsort()]
        sorted_time = sorted_data[:, 0]
        sorted_sigma = sorted_data[:, 1]
        return sorted_time, sorted_sigma
    
    def sorted_time(self):
        return self.sorted_data()[0]

    def sorted_sigma(self):
        return self.sorted_data()[1]

    def histogram(self):
        time = self.time
        sigma = self.sigma
        plt.hist(time, bins = len(time)/30, normed = True, weights = sigma, histtype = 'step', label = "Experimental Data")
        plt.xlabel("Decay time (ps)")
        plt.ylabel("Events")

    def fit_function(self, time, sigma, tau, a):
        """Return PDF of decay time."""
        # a = fraction of signal (real decays) in sample
        fit = a * ((1./(2.*tau)) * np.exp(sigma**2/(2.*tau**2.) - time/tau) * sp.erfc((1./np.sqrt(2.))*(sigma/tau - time/sigma))) + ((1 - a)*(1./(sigma * np.sqrt(2.*np.pi)) * np.exp((-1./2.)*time**2/sigma**2)))
        return fit
    
    def integrate_fit(self, tau, sigma):
        """Return integral of fit function to check that PDF is normalised."""
        f = self.fit_function(self.sorted_time(), sigma, tau, 1.)
        return np.trapz(f, self.sorted_time())
        
    def plot_fit_function(self, tau, a):
        """Plot fit function for all data on the same graph."""
        plt.plot(self.sorted_time(), self.fit_function(self.sorted_time(), self.sorted_sigma(), tau, a), '-', label = "PDF $\\tau$ = " + str(tau) + ")", markersize = 10)
        plt.xlabel("Decay time (ps)")
        plt.ylabel("Fit Function (PDF)")
        plt.title("Decay Time Distribution")
        plt.legend()
    
    def NLL(self, tau, a):
        """Return NLL of fit function."""
        fit = self.fit_function(self.time, self.sigma, tau, a)
        log = -np.sum(np.log(fit))
        return log
    
    def plot_NLL(self, x1, x2, a):
        x = np.arange(x1, x2, 0.0001)
        y = []
        for tau in x:
            y.append(self.NLL(tau, a))
        plt.xlabel("$\\tau$ (ps)")
        if a == 1:
            plt.plot(x, y, label = "NLL neglecting background")
        else:
            plt.plot(x, y, label = "NLL with background")
        plt.title("Negative Log Likelihood Function")
    
    def contour(self, x = np.arange(0.3, 0.5, 0.001), y = np.arange(0.7, 1., 0.001)):  
        """Plot contour of NLL as a function of a and tau."""
        z = []
        for a in y:
            for tau in x:
                z.append(self.NLL(tau, a))
        Y, X = np.meshgrid(x, y)
        z = np.array(z)
        z = z.reshape(len(y), len(x))
        plt.figure()  
        contours = plt.contour(Y, X, z, 20) 
        plt.xlabel("$\\tau$ (ps)")
        plt.ylabel("fraction of signal in the sample")
        plt.title("Contour Plot of NLL")
        plt.clabel(contours, fontsize=12)
        
    def plot_3D(self):
        """3D plot of NLL as a function of a and tau."""
        x = np.arange(0.3, 0.6, 0.01)
        y = np.arange(0.9, 1., 0.01)
        z = []
        for a in y:
            for tau in x:
                z.append(self.NLL(tau, a))
        Y, X = np.meshgrid(x, y)
        z = np.array(z)
        z = z.reshape(len(y), len(x))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
    
    def gradient(self, tau, a):
        """Return gradient of NLL."""
        h = 0.0001
        der_tau = (self.NLL(tau + h, a) - self.NLL(tau, a))/h 
        der_a = (self.NLL(tau, a + h) - self.NLL(tau, a))/h
        return np.array([der_tau, der_a])

    def inv_hessian(self, tau, a):
        """Return the inverse of the Hessian matrix of NLL."""
        h = 0.0001
        d2_dtau = (self.gradient(tau + h, a)[0] - self.gradient(tau, a)[0])/h
        d2_da = (self.gradient(tau, a + h)[1] - self.gradient(tau, a)[1])/h
        d2 = (self.gradient(tau, a + h)[0] - self.gradient(tau, a)[0])/h
        hessian = np.array([[d2_dtau, d2], [d2, d2_da]])
        inv_hessian = np.linalg.inv(hessian)
        return inv_hessian        

class Minimiser:
    def __init__(self, data, tau0, a0):
        self.data = data
        self.tau0 = tau0
        self.a0 = a0
    
    def lagrange_min(self, tau0, tau1, tau2, a = 1):
        """Return 3 points that bracket the minimum"""
        data = self.data
        f = data.NLL
        x = [tau0, tau1, tau2]
        y = [f(tau0, a), f(tau1, a), f(tau2, a)]
    
        num = (x[2]**2 - x[1]**2)*y[0] + (x[0]**2 - x[2]**2)*y[1] + (x[1]**2 - x[0]**2)*y[2]
        den = (x[2] - x[1])*y[0] + (x[0] - x[2])*y[1] + (x[1] - x[0])*y[2]
        x3 = 1./2.*(num/den)      
        return x3
    
    def parabolic(self, accuracy, a = 1):
        """Iterate lagrange_min to get closer to the minimum up to a given accuracy."""
        data = self.data
        tau0 = self.tau0 - 0.1
        tau1 = self.tau0
        tau2 = self.tau0 + 0.1
        f = data.NLL
        pairs = {f(tau0, a) : tau0, f(tau1, a) : tau1, f(tau2, a) : tau2}
        tau3 = self.lagrange_min(pairs.values()[0], pairs.values()[1], pairs.values()[2])
        tau = [tau0, tau1, tau2]
        if tau3 - min(pairs.values()) < accuracy:
            tau.append(tau3)
            #min_points = [tau[-3], tau[-2], tau[-1]]
            result = Parameters(tau, [a], self.data, "Parabolic Method")
            return result
    
        while (tau3) - min(pairs.values()) > accuracy:
            pairs.update({f(tau3, a) : tau3})
            y = pairs.keys()
            pairs.pop(max(y))
            tau3 = self.lagrange_min(pairs.values()[0], pairs.values()[1], pairs.values()[2])
            tau.append(tau3)
            #min_points = [tau[-3], tau[-2], tau[-1]] #only need last 3 points
        result = Parameters(tau, [a], self.data, "Parabolic Method")
        return result

    def grad_min(self):
        """Gradient method minimisation.
        
        Return instance of Parameters class.
        """
        data = self.data
        tau0 = self.tau0
        a0 = self.a0
        alpha = 1e-5
        x = np.array([np.full(10000, tau0), np.full(10000, a0)])
        x[:,1] = x[:,0] - alpha*data.gradient(tau0, a0)
        tau = []
        a = []
        for i in range(0, 100):
            x[:,i] = x[:,i-1] - alpha*data.gradient(x[0, i-1], x[1, i-1])
            if data.NLL(x[0,i], x[1,i]) < data.NLL(x[0,i-1], x[1, i-1]):
                tau.append(x[:, i][0])
                a.append(x[:, i][1])
            else: 
                break
        result = Parameters(tau, a, self.data, "Gradient Method")
        return result

    def newton(self):
        """Newton method minimisation.
        
        Return instance of Parameters class.
        """
        data = self.data
        tau0 = self.tau0
        a0 = self.a0
        x = np.array([np.full(10000, tau0), np.full(10000, a0)])
        x[:, 1] = x[:, 0] - np.dot(data.inv_hessian(tau0, a0), data.gradient(tau0, a0))
        tau = []
        a = []
        for i in range(0, 500):
            x[:,i] = x[:,i-1] - np.dot(data.inv_hessian(x[0, i-1], x[1, i-1]), data.gradient(x[0, i-1], x[1, i-1]))
            if data.NLL(x[0,i], x[1,i]) < data.NLL(x[0,i-1], x[1, i-1]):
                tau.append(x[:, i][0])
                a.append(x[:, i][1])
            else: 
                break
        result = Parameters(tau, a, self.data, "Newton Method")
        return result

    def quasi_newton(self):
        """Quasi-Newton method minimisation.
        
        Return instance of Parameters class.
        """
        data = self.data
        tau0 = self.tau0
        a0 = self.a0
        alpha = 1e-5
        x = np.array([np.full(10000, tau0), np.full(10000, a0)])
        x[:,1] = x[:,0] - alpha*data.gradient(tau0, a0)
        delta = np.full((2, 10000), 0)
        gamma = np.full((2, 10000), 0)
        delta[:, 1] = x[:, 1] - x[:, 0]
        gamma[:, 1] = data.gradient(x[0, 1], x[1, 1]) - data.gradient(x[0, 0], x[1, 0])
        G = 10000*[np.identity(2)]
        tau = []
        a = []
        for i in range(1, 100):
            delta[:, i-1] = x[:, i] - x[:, i-1]
            gamma[:, i-1] = data.gradient(x[0, i], x[1, i]) - data.gradient(x[0, i-1], x[1, i-1])
            G[i] = G[i-1] + np.outer(delta[:, i-1], delta[:, i-1])/np.dot(gamma[:, i-1], delta[:,i-1]) - (G[i-1]*np.outer(delta[:, i-1], delta[:, i-1]) * G[i-1-1])/np.dot((gamma[:, i-1]*G[i-1]), gamma[:, i-1])
            x[:,i] = x[:,i-1] - alpha*np.dot(G[i], data.gradient(x[0, i-1], x[1, i-1]))
            if data.NLL(x[0,i], x[1,i]) < data.NLL(x[0,i-1], x[1, i-1]):
                tau.append(x[:, i][0])
                a.append(x[:, i][1])
            else: 
                break
        result = Parameters(tau, a, self.data, "Quasi-Newton Method")
        return result

class Parameters:
    """Class for the parameters tau and a obtained through the minimisation."""
    
    def __init__(self, tau, a, data, label):
        self.label = label
        self.tau = tau
        self.a = a
        self.data = data
    
    def minimum(self):
        """Return parameters tau and a."""
        min_tau = self.tau[-1]
        min_a = self.a[-1]
        return min_tau, min_a
        
    def deviation_curv(self): 
        """Calculate standard deviation of tau from curvature of parabola when a = 1."""
        data = self.data
        tau0 = self.tau[-3]
        tau1 = self.tau[-2]
        tau2 = self.tau[-1]
        y0 = data.NLL(tau0, 1)
        y1 = data.NLL(tau1, 1)
        y2 = data.NLL(tau2, 1)
        curv = (2./((tau1-tau0)*(tau2-tau0)*(tau2-tau1))) * ((tau0*(y1-y2) + tau1*(y2-y0) + tau2*(y0-y1)))
        #return 1./np.sqrt(curv)
        h = 0.0001
        d2_dtau = (data.gradient(tau2 + h, 1)[0] - data.gradient(tau2, 1)[0])/h
        return 1./np.sqrt(curv), 1./np.sqrt(d2_dtau)
        
    def deviation_tau(self):
        """Calculate (underestimation of) standard deviation of tau as value at which NLL = NLL + 0.5.
        
        Use Newton-Raphson method to find root
        """
        data = self.data
        tau = self.minimum()[0]
        a = self.minimum()[1]
        x1 = np.full(20, 0.3)
        for i in range(20):
            x1[i] = x1[i-1] - (data.NLL(x1[i-1], a) - data.NLL(tau, a) - 0.5)/(data.gradient(x1[i-1], a)[0])
        x2 = np.full(20, 0.5)
        for i in range(20):
            x2[i] = x2[i-1] - (data.NLL(x2[i-1], a) - data.NLL(tau, a) - 0.5)/(data.gradient(x2[i-1], a)[0] )
        return x1[-1] - tau, x2[-1] - tau
    
    def deviation_a(self):
        """Calculate (underestimation of) standard deviation of a as value at which NLL = NLL + 0.5.
        
        Use Newton-Raphson method to find root
        """
        data = self.data
        tau = self.minimum()[0]
        a = self.minimum()[1]
        x1 = np.full(100, 0.8)
        for i in range(100):
            x1[i] = x1[i-1] - (data.NLL(tau, x1[i-1]) - data.NLL(tau, a) - 0.5)/(data.gradient(tau, x1[i-1])[1])
        x2 = np.full(100, 0.99)
        for i in range(100):
            x2[i] = x2[i-1] - (data.NLL(tau, x2[i-1]) - data.NLL(tau, a) - 0.5)/(data.gradient(tau, x2[i-1])[1])
        return x1[-1] - a, x2[-1] - a
    
    def covariance(self):
        """Return covariance matrix of NLL at minimum point and standard deviation as square root of diagonal elements."""
        data = self.data
        sigma_tau = np.sqrt(np.diag(data.inv_hessian(self.minimum()[0], self.minimum()[1]))[0])
        sigma_a = np.sqrt(np.diag(data.inv_hessian(self.minimum()[0], self.minimum()[1]))[1])
        correlation = data.inv_hessian(self.minimum()[0], self.minimum()[1])[0][1]/(sigma_tau * sigma_a)
        return data.inv_hessian(self.minimum()[0], self.minimum()[0]), sigma_tau, sigma_a, correlation
        
    def ellipse(self):
        data = self.data
        tau = self.minimum()[0]
        a = self.minimum()[1]
        taus = []
        x1 = np.full(100, 0.3)
        a_s = []
        for j in np.arange(a - 0.0087, a + 0.0084, 0.001):
            for i in range(100):
                x1[i] = x1[i-1] - (data.NLL(x1[i-1], j) - data.NLL(tau, a) - 0.5)/(data.gradient(x1[i-1], j)[0])
            if np.isnan(x1[-1]) == False:
                a_s.append(j)
                taus.append(x1[-1]) 
                
        x2 = np.full(100, 0.45)
        a_s1 = []
        point1 = []
        for j in np.arange(a - 0.0087, a + 0.0085, 0.001):
            for i in range(100):
                x2[i] = x2[i-1] - (data.NLL(x2[i-1], j) - data.NLL(tau, a) - 0.5)/(data.gradient(x2[i-1], j)[0])
            if np.isnan(x2[-1]) == False:
                a_s1.append(j)
                point1.append(x2[-1]) 
        plt.figure()
        plt.title("Standard Error Ellipse")
        plt.plot(taus, a_s, 'b.', markersize = 5)
        plt.plot(point1, a_s1, 'b.', markersize = 5)
        plt.scatter(self.minimum()[0], self.minimum()[1], color = 'r')
        plt.xlabel("$\\tau$ (ps)")
        plt.ylabel("a")
        plt.legend()
        plt.xlim(0.403, 0.416)
        plt.show()
        return "Positive sigma tau", np.amax(point1) - self.minimum()[0], "Positive sigma a", np.amax(a_s1) - self.minimum()[1], "Negative sigma tau", np.amin(taus) - self.minimum()[0], "Negative sigma a", np.amin(a_s) - self.minimum()[1]
    
    def plot_minimisation(self):
        """Plot result of minimisation."""
        data = self.data
        if self.a != [1]:
            plt.figure()
            data.contour(np.arange(0.4, 0.44, 0.001), np.arange(0.9, 1., 0.001))
            plt.plot(self.minimum()[0], self.minimum()[1], 'r.')
            plt.plot(self.tau, self.a, label = self.label + "\nMinimisation")
            plt.errorbar(self.minimum()[0], self.minimum()[1], xerr = self.covariance()[1], yerr = self.covariance()[2])
            #plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
            plt.figure()
            data.plot_NLL(0.25, 0.55, self.minimum()[1])
            data.plot_NLL(0.25, 0.55, 1)
            plt.xlim(0.25, 0.55)
            plt.legend(loc = 9)
            plt.show()
       
        if self.a == [1]:
            #Plot of NLL and its parabolic approximation (starting and final point)
            x0 = self.tau[0]
            x1 = self.tau[1]
            x2 = self.tau[2]
            x = np.arange(0.25, 0.55, 0.001)
            y = ((x-x1)*(x-x2)*data.NLL(x0, 1))/((x0-x1)*(x0-x2)) + ((x-x0)*(x-x2)*data.NLL(x1, 1))/((x1-x0)*(x1-x2)) + ((x-x0)*(x-x1)*data.NLL(x2, 1))/((x2-x0)*(x2-x1))
            data.plot_NLL(0.25, 0.55, 1)
            plt.plot(x, y, 'r--', label = "Parabolic Approximation")
            plt.annotate("a", xy = (x0, data.NLL(x0, 1)), xytext = (x0, 6800), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            plt.annotate("b", xy = (x1, data.NLL(x1, 1)), xytext = (x1, 6700), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            plt.annotate("c", xy = (x2, data.NLL(x2, 1)), xytext = (x2, 6800), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            x0 = self.tau[-1]
            x1 = self.tau[-2]
            x2 = self.tau[-3]
            y = ((x-x1)*(x-x2)*data.NLL(x0, 1))/((x0-x1)*(x0-x2)) + ((x-x0)*(x-x2)*data.NLL(x1, 1))/((x1-x0)*(x1-x2)) + ((x-x0)*(x-x1)*data.NLL(x2, 1))/((x2-x0)*(x2-x1))
            plt.plot(x, y, 'b--')
            plt.xlim(0.25, 0.55)
            plt.legend(loc = 9)
            #Plot of NLL at minimum with negative and positive standard deviations
            plt.figure()
            point1 = self.minimum()[0] + self.deviation_tau()[0]
            point2 = self.minimum()[0] + self.deviation_tau()[1]
            point3 = self.minimum()[0]
            plt.scatter((point1, point2, point3), (data.NLL(point1, 1.), data.NLL(point2, 1.), data.NLL(point3, 1.)))
            plt.annotate("$\\Delta$ NLL = 0.5", xy = (point2 + 0.003, data.NLL(point1, 1)), xytext = (0.391, data.NLL(point1, 1)), arrowprops=dict(arrowstyle="-"))
            data.plot_NLL(self.minimum()[0]-0.01, self.minimum()[0]+0.01, 1.)
            plt.ylabel("NLL")
            #Plot of histogram of data and fit function with optimised parameters
            plt.figure()
            data.histogram()
            data.sigma = 0.35
            plt.plot(data.sorted_time(), data.fit_function(data.sorted_time(), data.sigma ,self.minimum()[0], self.minimum()[1]), color = 'red', label = "PDF ($\\sigma$) = " + str(data.sigma))
            plt.legend()            
            plt.show()