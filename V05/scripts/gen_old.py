# Importanweisungen

import sys
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

def exponential(x, c, y0,x0):
    return np.exp(c * (x-x0)) * y0

def sym_exponential(x, c, y0,x0):
    return np.exp(c * np.abs(x-x0)) * y0

def double_exponential(x, c1,c2, y0,x0):
    return np.exp(-c1* (x-x0)) * y0*np.heaviside(x-x0,1)+ np.heaviside(x0-x,1)*np.exp(c2* (x-x0)) * y0

def custom(x,x0,A,d):
    return A * np.exp(-(x - x0)**2 / 2 / d**2)

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

# fittet ein dataset mit gegebenen x und y werten, eine funktion und ggf. anfangswerten und y-Fehler
# gibt die passenden parameter der funktion, sowie dessen unsicherheiten zurueck
#
# https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i#
# Updated on 4/6/2016
# User: https://stackoverflow.com/users/1476240/pedro-m-duarte
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
unc_w = 0.3

#calc zeitkalibrier
awkdata = np.loadtxt("V05/data/awk.out", skiprows = 0)
adata = unp.uarray(np.diff(awkdata),0)
#print(adata)
print(unv(np.mean(adata)))
print ("%s:%s"%("Zeitkalibrier",mean(adata)))
#XRD
unc_x = 0.002/math.sqrt(3)*0
unc_y = 0.005/math.sqrt(3)*0
typ = [ "Zeitdifferenzen","Zeitkalibrierung","Positronium_Zeitdifferenz","Energiespektrum_Start", "Energiespektrum_Stop", ]
position = 39.76
kali = 0.64/514.8
width = 1 # == 1Grad
for t in typ:
    print(t)
    data = np.genfromtxt("V05/data/%s_cut.Spe"%(t))
    ydata = unp.uarray(data[:],np.sqrt(data[:]))
    xdata = np.linspace(0,8191,8192)
    #plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=1, label='Messpunkte')
    if(t=="Zeitkalibrierung"):
        plt.bar(unv(xdata), unv(ydata), width=width*10, color='r', yerr=usd(0), label= 'Messpunkte')
    elif(t=="Positronium_Zeitdifferenz"):
        xdata = xdata*kali
        plt.bar(unv(xdata[1400:1700]), unv(ydata[1400:1700]), width=width*kali, color='r', yerr=usd(ydata[1400:1700]), label= 'Messpunkte')

        fit = fit_curvefit2(unv(xdata[1400:1700]), unv(ydata[1400:1700]), custom, p0 = [1471*kali, 475,17*kali])

        xfit = np.linspace(1400, 1700, 400)
        xfit = xfit*kali
        yfit = custom(xfit, *unv(fit))
        plt.plot(unv(xfit), unv(yfit), color = 'blue',linewidth=2, label='Gauss Fit\n$T_0$=%s\n$N$=%s\n$\Delta T$=%s'%tuple(fit))

        plt.legend(prop={'size':fig_legendsize})
        plt.grid()
        plt.tick_params(labelsize=fig_labelsize)
        plt.xlabel('Zeitdifferenz in $\mu$s')
        plt.ylabel('Ereignisse')
        plt.savefig(("V05/img/%s_zoom"%(t)).replace(".",",") + ".pdf")
        plt.show()

        plt.bar(unv(xdata), unv(ydata), width=width*kali, color='r', yerr=usd(ydata), label= 'Messpunkte')

    elif(t=="Energiespektrum_Stop" or t=="Energiespektrum_Start"):
        plt.bar(unv(xdata), unv(ydata), width=width, color='r', yerr=usd(ydata), label= 'Messpunkte')
    elif(t=="Zeitdifferenzen"):

        xdata = xdata*kali
        ww = 1300
        hh = 1600
        plt.bar(unv(xdata[ww:hh]), unv(ydata[ww:hh]), width=width*kali, color='r', yerr=usd(ydata[ww:hh]), label= 'Messpunkte')

        plt.axhline(y=125,color="green")
        plt.axhline(y=550,color="y")
        plt.axvline(x=1.75,color="y")
        plt.axhline(y=337,color="m")
        plt.axvline(x=1.9,color="m")

        plt.legend(prop={'size':fig_legendsize})
        plt.grid()
        plt.tick_params(labelsize=fig_labelsize)
        plt.xlabel('Zeitdifferenz in $\mu$s')
        plt.ylabel('Ereignisse')
        plt.savefig(("V05/img/%s_zoom"%(t)).replace(".",",") + ".pdf")
        plt.show()

        plt.bar(unv(xdata), unv(ydata), width=width*kali, color='r', yerr=usd(ydata), label= 'Messpunkte')
        m =mean(ydata[2000:])
        print("%s:%s"%("Untergrund",m))

        fit = fit_curvefit2(unv(xdata), unv(ydata-m),double_exponential,yerr=usd(ydata-m), p0 = [4.5,4.5,447,1.74])
        print(fit)
        print(1/fit)
        print(1/fit*np.log(2))
        xfit = np.linspace(0,8000,8001)
        xfit = xfit*kali
        yfit = double_exponential(xfit, *unv(fit))

        plt.plot(unv(xfit), unv(yfit+m), color = 'm',linewidth=1, label='DExp Fit\n$\\lambda_1$=%s\n$\\lambda_2$=%s\n$N$=%s\n$T_0$=%s'%tuple(fit))

        plt.axhline(y=unv(mean(ydata[2500:])),color="g")
    else:
        xdata = xdata*kali
        plt.bar(unv(xdata), unv(ydata), width=width*kali, color='r', yerr=usd(ydata), label= 'Messpunkte')

    plt.legend(prop={'size':fig_legendsize})
    plt.grid()
    plt.tick_params(labelsize=fig_labelsize)
    if(t=="Zeitkalibrierung"):
        plt.xlabel('Kanal')
    elif(t=="Energiespektrum_Stop" or t=="Energiespektrum_Start"):
        plt.xlabel('Energie in a.u.')
    else:
        plt.xlabel('Zeitdifferenz in $\mu$s')
    plt.ylabel('Ereignisse')
    plt.savefig(("V05/img/%s"%(t)).replace(".",",") + ".pdf")
    plt.show()


#end
