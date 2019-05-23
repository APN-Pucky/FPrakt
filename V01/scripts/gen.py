# Importanweisungen

import numpy as np
import statistics as stat
import scipy as sci
import scipy.fftpack
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.axes as axes
from matplotlib import colors as mcolors
import math
from scipy import optimize
import uncertainties as unc
import uncertainties.unumpy as unp
import uncertainties.umath as umath
import glob
import os
unv=unp.nominal_values
usd=unp.std_devs


# Konstanten fuer einheitliche Darstellung

fig_size = (10, 6)
fig_legendsize = 14
fig_labelsize = 12
matplotlib.rcParams.update({'font.size': fig_labelsize})

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#colors

# mathe Funktionen

def mean(n):
    # find the mean value and add uncertainties
    k = np.mean(n)
    err = stat.variance(unv(n))
    return unc.ufloat(unv(k), math.sqrt(usd(k)**2 + err))

def fft(y):
    N = len(y)
    fft = scipy.fftpack.fft(y)
    return 2 * abs(fft[:N//2]) / N

    # allgemeine Fitfunktionen

def linear(x,m): # lineare Funktion mit f(x) = m * x
    return(m*x)

def gerade(x, m, b): # gerade mit = f(x) = m * x + b
    return (m*x + b)

def cyclic(x, a, f, phi):
    return a * np.sin(x * f - phi)

def cyclicOff(x, a, f, phi, offset):
    return cyclic(x, a, f, phi) + offset

def gauss(x, x0, A, d, y0):
    return A * np.exp(-(x - x0)**2 / 2 / d**2) + y0

def exponential(x, c, y0):
    return np.exp(c * x) * y0

# fittet ein dataset mit gegebenen x und y werten, eine funktion und ggf. anfangswerten und y-Fehler
# gibt die passenden parameter der funktion, sowie dessen unsicherheiten zurueck
#
# https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i#
# Updated on 4/6/2016
# User: https://stackoverflow.com/users/1476240/pedro-m-duarte
def fit_curvefit(datax, datay, function, p0=None, yerr=None, **kwargs):
    pfit, pcov = \
         optimize.curve_fit(function,datax,datay,p0=p0,\
                            sigma=yerr, epsfcn=0.0001, **kwargs)
    error = []
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_curvefit = pfit
    perr_curvefit = np.array(error)
    return pfit_curvefit, perr_curvefit

# usage zB:
# pfit, perr = fit_curvefit(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
# fuer eine gerade mit anfangswerten m = 1, b = 0

# weitere Werte, Konstanten
# Werte von https://physics.nist.gov/cuu/Constants/index.html[0]

c = 299792458 # m/s
k_B = unc.ufloat_fromstr("1.38064852(79)e-23") # J K-1 [0]
h = unc.ufloat_fromstr("4.135667662(25)e-15") # eV s [0]
r_e = unc.ufloat_fromstr("2.8179403227(19)e-15") # m [0]
R = unc.ufloat_fromstr("8.3144598(48)") # J mol-1 K-1 [0]
K = 273.15 # kelvin
g = 9.81 # m/s^2
rad = 360 / 2 / math.pi
grad = 1/rad
# Unsicherheiten

unc_n = 0
unc_p = 0
# import der messwerte
names = glob.glob("V01/fit/*.dat")
iter= -1
peakid1 = 0
peakid2 =  0
for name in names:
    iter +=1
    data = np.loadtxt(name, skiprows = 0, delimiter = " ")

    nname =os.path.basename(name)
    print(nname)
    if(nname=="NaNa.dat"):
        peakid1 = 1
        peakid2 = 5
    if(nname=="NaGe.dat"):
        peakid1 = 1
        peakid2 = 2
    xdata = unp.uarray(data[:,0],unc_n)
    ydata = unp.uarray(data[:,1],unc_p)
    model = unp.uarray(data[:,-2],unc_p)
    residual = unp.uarray(data[:,-1],unc_p)
    peak1 = unp.uarray(data[:,peakid1+1],unc_p)
    peak2 = unp.uarray(data[:,peakid2+1],unc_p)

    ybackground = model - peak1 - peak2
    limit = 0.4
    xpeak1, ypeak1 = zip(*((x, y) for x, y in zip(xdata, peak1) if y > limit))
    xpeak2, ypeak2 = zip(*((x, y) for x, y in zip(xdata, peak2) if y > limit))
    xmodel, ymodel = zip(*((x, y) for x, y in zip(xdata, model) if y > limit))
    xback, yback = zip(*((x, y) for x, y in zip(xdata, ybackground) if y > limit))


    fig=plt.figure(figsize=fig_size)

    ## Plot
    frame1=fig.add_axes((.1,.3,.8,.6))
    #plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=2,label='Druck')
    plt.plot(unv(xdata), unv(ydata), '.',label='Messung',linewidth='1')
    plt.plot(unv(xback), unv(yback), label='background',linewidth='1')
    plt.plot(unv(xmodel), unv(ymodel), label='model',linewidth='1')
    plt.plot(unv(xpeak1), unv(ypeak1), label='peak',linewidth='1')
    plt.plot(unv(xpeak2), unv(ypeak2), label='peak',linewidth='1')

    plt.gca().set_yscale('log');
    plt.legend(prop={'size':fig_legendsize})
    plt.grid()
    plt.ylabel('Ereignisse')
    ## Residual
    frame2=fig.add_axes((.1,.1,.8,.2))
    plt.plot(unv(xdata), unv(residual), '.',linewidth='1')

    plt.ylabel('weighted residuals')
    plt.grid()
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Kanal')
    plt.savefig("EDX/images/" + nname.split('.')[0] + ".pdf")
    plt.show()

#end
