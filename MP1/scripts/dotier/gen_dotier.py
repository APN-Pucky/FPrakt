# Importanweisungen

import os
import re
import numpy as np
import statistics as stat
import scipy as sci
import scipy.fftpack
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.axes as axes
import matplotlib.patches as patches
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

def exponential(x, c, y0,yy):
    return np.exp(c * x) * y0-yy

def custom(x,y,c):
    return 1/((x-0.066)*c)+y

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
    return unp.uarray(pfit_curvefit, perr_curvefit)

# fittet ein dataset mit gegebenen x und y werten, eine funktion und ggf. anfangswerten und y-Fehler
# gibt die passenden parameter der funktion, sowie dessen unsicherheiten zurueck
#
# https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i#
# Updated on 4/6/2016
# User: https://stackoverflow.com/users/1476240/pedro-m-duarte
def fit_curvefit2(datax, datay, function, p0=None, yerr=None, **kwargs):
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
    return unp.uarray(pfit_curvefit, perr_curvefit)
# usage zB:
# pfit, perr = fit_curvefit(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
# fuer eine gerade mit anfangswerten m = 1, b = 0

# weitere Werte, Konstanten
# Werte von https://physics.nist.gov/cuu/Constants/index.html[0]

c = 299792458 # m/s
e = unc.ufloat_fromstr("1.6021766208(98)e-19") # C
k_B = unc.ufloat_fromstr("1.38064852(79)e-23") # J K-1 [0]
h = unc.ufloat_fromstr("4.135667662(25)e-15") # eV s [0]
r_e = unc.ufloat_fromstr("2.8179403227(19)e-15") # m [0]
R = unc.ufloat_fromstr("8.3144598(48)") # J mol-1 K-1 [0]
K = 273.15 # kelvin
g = 9.81 # m/s^2
rad = 360 / 2 / math.pi
grad = 1/rad
# Unsicherheiten
unc_x = 0.002/math.sqrt(3)
unc_y = 0.005/math.sqrt(3)
unc_w = 0.3
# import der messwerte
for fname in os.listdir("MP1/data/dotier/"):
   with open("MP1/data/dotier/" + fname) as f:
       lines = (line for line in f if not line.startswith('#'))
       names = next(lines,None).split(";")
       factors = next(lines,None).split(";")
       uncc = next(lines,None).split(";")
       m = re.compile(".*\\((\w+)\\)")
       units = [m.findall(names[0])[0],m.findall(names[1])[0]]
       data = np.genfromtxt(lines,  delimiter = ";")

   fname = fname.split(".")[0]
   if fname=="widerstand":
       for i in range(6):
           xdata = unp.uarray(data[:,2*i+0]*float(factors[2*i+0]),float(uncc[2*i+0])*float(factors[2*i+0])/2/math.sqrt(3))
           ydata = unp.uarray(data[:,2*i+1]*float(factors[2*i+1]),float(uncc[2*i+1])*float(factors[2*i+1])/2/math.sqrt(3))
           mx = mean(xdata[~np.isnan(unv(xdata))])
           my = mean(ydata[~np.isnan(unv(ydata))])
           print(i+1, " Probe")
           print("R=",mx)
           print("d=",my)
           rho = mx*my*math.pi/math.log(2)
           if i==4:
               rho = rho*0.83
           if i==3:
               rho = rho*0.84
           print("RHO=",rho/10)
           mn = 1350
           mp = 480
           c=2.1e19
           n = (-umath.sqrt((1/(2*e*mn*rho/10)**2)-c*mp/mn) + 1/(2*e*mn*rho/10))
           p = (+umath.sqrt((1/(2*e*mp*rho/10)**2)-c*mn/mp) + 1/(2*e*mp*rho/10))
           print("pn=",c/p)
           print("nn=",n)
           print("pp=",p)
           #print("np=",c/n)
           print("n_i=",1/(rho/10*e*(3900+1900)))
   if fname == "polier":
       xxdata = unp.uarray([],[])
       yydata = unp.uarray([],[])
       for i in range(19):
           ydata = unp.uarray(data[:,2*i+0]*float(factors[2*i+0]),float(uncc[2*i+0])*float(factors[2*i+0])/2/math.sqrt(3))
           xdata = unp.uarray(data[:,2*i+1]*float(factors[2*i+1]),float(uncc[2*i+1])*float(factors[2*i+1])/2/math.sqrt(3))
           my = mean(ydata[~np.isnan(unv(ydata))])
           mx = xdata[~np.isnan(unv(xdata))][0]
           #rho = mx*my*math.pi/math.log(2)
           yydata =np.append(yydata,my)
           xxdata = np.append(xxdata,mx)
       yyydata = unp.uarray([],[])
       print(xxdata)
       ii = 18
       for i in range(ii):
           rrho = math.log(2)/math.pi*(-1)*(((yydata[i]-yydata[i+1])))/(xxdata[i+1]-xxdata[i])/(yydata[i+1]*yydata[i])
           yyydata = np.append(yyydata,rrho)

       fig=plt.figure(figsize=fig_size)
       ax = fig.gca()
       print(yydata)
       #ax.errorbar(unv(xxdata[0:ii]),unv(yyydata), usd(yyydata), usd(xxdata[0:ii]),fmt=' ', capsize=5,linewidth=2, label="Polier")
       ax.errorbar(unv(xxdata),unv(yydata), usd(yydata), usd(xxdata),fmt=' ', capsize=5,linewidth=2, label="Messung")
       #ax.errorbar(unv(xxdata[3:-1]),unv(1/yydata[3:-1]), usd(1/yydata[3:-1]), usd(xxdata[3:-1]),fmt=' ', capsize=5,linewidth=2, label="Polier")
       yyyydata = -math.log(2)/math.pi*np.diff(1/yydata)/np.diff(xxdata)
       ax.set_yscale("log", nonposy='clip')
       plt.legend(prop={'size':fig_legendsize})
       plt.grid()
       plt.tick_params(labelsize=fig_labelsize)
       plt.xlabel("Schichtenabtrag (mm)")
       plt.ylabel(names[0])
       plt.savefig("MP1/img/%s.pdf"%(fname+"_wdst"))
       plt.show()

       fig2=plt.figure(figsize=fig_size)
       ax2 = fig2.gca()
       mn = 1350
       mp = 480
       c=2.1e19
       yyyydata = (+((1/(2*e*mp*1/yyyydata/10)**2)-c*mn/mp)**0.5 + 1/(2*e*mp*1/yyyydata/10))
       print(yyyydata)
       ax2.errorbar(unv(xxdata[0:18]),unv(yyyydata), usd(yyyydata), usd(xxdata[0:18]),fmt=' ', capsize=5,linewidth=2, label="Messung")
       ax2.set_yscale("log", nonposy='clip')

       print("mean p :",mean(yyyydata[-3:-1]))


       plt.legend(prop={'size':fig_legendsize})
       plt.grid()
       plt.tick_params(labelsize=fig_labelsize)
       plt.xlabel("Schichtenabtrag (mm)")
       plt.ylabel("Löcherkonzentration (cm$^{-3}$)")
       plt.savefig("MP1/img/%s.pdf"%(fname+"_konz"))
       plt.show()

       fig2=plt.figure(figsize=fig_size)
       ax2 = fig2.gca()
       mn = 1350
       mp = 480
       c=2.1e19
       yyyydata = yyyydata-0.9e14
       print(yyyydata)
       ax2.errorbar(unv(xxdata[0:15]),unv(yyyydata[0:15]), usd(yyyydata[0:15]), usd(xxdata[0:15]),fmt=' ', capsize=5,linewidth=2, label="Messung")
       ax2.set_yscale("log", nonposy='clip')

       print("mean p :",mean(yyyydata[-3:-1]))


       plt.legend(prop={'size':fig_legendsize})
       plt.grid()
       plt.tick_params(labelsize=fig_labelsize)
       plt.xlabel("Schichtenabtrag (mm)")
       plt.ylabel("Löcherkonzentration (cm$^{-3}$)")
       plt.savefig("MP1/img/%s.pdf"%(fname+"_konz_2"))
       plt.show()








#end
