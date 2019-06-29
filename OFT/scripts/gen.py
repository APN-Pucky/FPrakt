# Importanweisungen
import matplotlib.patches as mpatches
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
fig_labelsize = 16
matplotlib.rcParams.update({'font.size': fig_labelsize})

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#colors

# mathe Funktionen

# mathe Funktionen
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def find_nearest(array, value):
    array[find_nearest_index(array,value)]
def normalize(ydata):
   return (ydata-np.amin(ydata))/(np.amax(ydata)-np.amin(ydata))
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


def fit_curvefit2(datax, datay, function, p0=None, yerr=None, **kwargs):
    pfit, pcov = \
         optimize.curve_fit(function,datax,datay,p0=p0,\
                            sigma=yerr, epsfcn=0.0001, **kwargs, maxfev=1000000)
    error = []
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_curvefit = pfit
    perr_curvefit = np.array(error)
    return unp.uarray(pfit_curvefit, perr_curvefit)

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

# %% histo pyplot
names = glob.glob("OFT/data/1/*.txt")
for name in names:
    data = np.loadtxt(name, skiprows = 4, usecols=(0,1), delimiter = ";")
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]

    fig=plt.figure(figsize=fig_size)
    plt.plot(data[:,0]*1000, data[:,1], '-')

    plt.gca().set_yscale('log');
    #plt.gca().set_xscale('log');
    #plt.legend(prop={'size':fig_legendsize})
    plt.grid()
    plt.ylabel('Intensität in a.u.')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Position in mm')
    plt.savefig("OFT/img/1/%s"%(nnname + ".png"))
    plt.show()

# %% fit pyplot
names = glob.glob("OFT/data/2/*.txt")
for name in names:
    data = np.loadtxt(name, skiprows = 4, usecols=(0,1), delimiter = ";")
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]

    fig=plt.figure(figsize=fig_size)
    plt.plot(data[:,0]*1000, data[:,1], '-')

    if nnname=="2_gitter_g5":
        xs = [2.5,9,16,23,30.5,37.5,45,52,60,67.5,75,82.5,90,98,105]
        center = 7
    for x in xs:
        fig.gca().axvline(x=x,ymin=0,ymax=1, color='r')

    plt.gca().set_yscale('log');
    #plt.gca().set_xscale('log');
    #plt.legend(prop={'size':fig_legendsize})
    plt.grid()
    plt.ylabel('Intensität in a.u.')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Position in mm')
    plt.savefig("OFT/img/2/%s"%(nnname + ".png"))
    plt.show()

    fig=plt.figure(figsize=fig_size)
    xd = np.linspace(-center,len(xs)-center-1,len(xs))
    yrr = []
    for k in range(len(xs)):
        yrr.append(1)
    plt.errorbar(xd,xs,yerr=yrr, fmt='x',capsize=5, label="Peaks",color='r')

    fit = fit_curvefit2(xd,xs,gerade,yerr=yrr)
    xfit = np.linspace(xd[0],xd[-1],4000)
    yfit = gerade(xfit, *unv(fit))
    plt.plot(unv(xfit), unv(yfit), color = 'green',linewidth=2, label='Linear Fit f=ax+b\na=%s,\nb=%s'%(fit[0],fit[1]))


    plt.grid()
    plt.legend(prop={'size':fig_legendsize})
    plt.ylabel('Position in mm')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Peaknummer')
    plt.savefig("OFT/img/2/%s"%(nnname + ".png"))
    plt.show()
