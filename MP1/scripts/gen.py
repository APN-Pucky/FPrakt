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

def exponential(x, c, y0):
    return np.exp(c * x) * y0

def custom(x,I0,IP,a):
    return I0*(np.exp(x*a)-1)-IP

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
for fname in os.listdir("MP1/data/"):
   with open("MP1/data/" + fname) as f:
       lines = (line for line in f if not line.startswith('#'))
       names = next(lines,None).split(";")
       m = re.compile(".*\\((\w+)\\)")
       units = [m.findall(names[0])[0],m.findall(names[1])[0]]
       uncc = next(lines,None).split(";")
       data = np.loadtxt(lines, skiprows = 0, delimiter = ";")
   fname = fname.split(".")[0]

   ydata = unp.uarray(data[:,0],float(uncc[0])/2/math.sqrt(3))
   xdata = unp.uarray(data[:,1],float(uncc[1])/2/math.sqrt(3))


   fig=plt.figure(figsize=fig_size)

   ax = fig.gca()
   ax.axhline(y=0, color='k',linewidth=1)
   ax.axvline(x=0, color='k',linewidth=1)
   #xdata[i] = unc.ufloat(unv(xdata[i]),usd(xdata[i])*2)
   color = next(ax._get_lines.prop_cycler)['color']
   ax.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', color=color,capsize=5,linewidth=2, label=fname.replace("_"," ").replace("t ","t T=")+"°C")
   T = float(fname.split("_")[2])+K
   if fname.split("_")[0]=="PolykristallineZelle":
       n= 13
   else:
       n = 5
   pfit = fit_curvefit2(unv(xdata), unv(ydata), custom, yerr = usd(ydata),maxfev=100000, p0 = [np.amin(unv(ydata)), unv(ydata[find_nearest_index(xdata,0)]),unv(e/k_B/T/n)])

   xfit = np.linspace(-1, 1)
   yfit = custom(xfit, unv(pfit[0]),unv(pfit[1]),unv(pfit[2]))
   color = next(ax._get_lines.prop_cycler)['color']
   ax.plot(unv(xfit), unv(yfit),color="orange",linewidth=1)
   color = next(ax._get_lines.prop_cycler)['color']


   plt.errorbar([], [],[],[], ' ', color="orange",label='Diodenkennlinien Fit')
   print(pfit)
   if fname.split("_")[1]=="unbeleuchtet" and fname.split("_")[0]=="PolykristallineZelle":

       #pfit = fit_curvefit2(unv(xdata[0:-2]), unv(ydata[0:-2]), custom, yerr = usd(ydata[0:-2]),maxfev=100000, p0 = [np.amin(unv(ydata)), unv(ydata[find_nearest_index(xdata,0)]),unv(e/k_B/T/n)])

       xfit = np.linspace(-1, 1)
       yfit = custom(xfit, 0.33547,0.27644,7.12)
       ax.plot(unv(xfit), unv(yfit),color="violet",linewidth=1)
       plt.errorbar([], [],[],[], ' ', color="violet",label='Diodenkennlinien Fit ohne Ausreißer')

   if fname.split("_")[1]!="unbeleuchtet":
       uoc=umath.log(pfit[1]/pfit[0]+1)/pfit[2]
       xx = np.linspace(0,1);
       ind = np.argmin(xdata*ydata)
       yind = find_nearest_index(yfit,0)
       xind = find_nearest_index(xdata,0)
       ax.add_patch(patches.Rectangle((0,0),unv(xfit[yind]),unv(ydata[xind]),facecolor=color))
       ax.add_patch(patches.Rectangle((0,0),unv(xdata[ind]),unv(ydata[ind]),facecolor="red"))

       plt.errorbar([], [],[],[], ' ', color="green",label='$I_{sc} = %s$ %s' % (ydata[xind],units[0]))
       plt.errorbar([], [],[],[], ' ', color="green", label='$U_{oc} = %s$ %s' % (uoc,units[1]))
       plt.errorbar([], [],[],[], ' ', color="red",label='$I_{MPP} = %s$ %s' % (ydata[ind],units[0]))
       plt.errorbar([], [],[],[], ' ', color="red", label='$U_{MPP} = %s$ %s' % (xdata[ind],units[1]))
       plt.errorbar([], [],[],[], ' ', color="red", label='$P_{MPP} = %s$ %s%s' % (xdata[ind]*ydata[ind],units[0],units[1]))
       plt.errorbar([], [],[],[], ' ', color="white", label='$FF = %s$ %s' % (xdata[ind]*ydata[ind]/(ydata[xind]*xfit[yind])*100,"%"))




   #print(np.amin(xdata*ydata))
   #print(np.amin(xfit*yfit))
   #ff=np.amin(xfit*yfit)
   #pp=np.amin(xdata*ydata)
   #print(mean([ff,pp]))
   #print()

   #pfit, perr = fit_curvefit(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
   #pp = unp.uarray(pfit, perr)
   #xdata = np.linspace(unv(xdata[0]),unv(xdata[-1]))
   #plt.plot(xdata,unv(gerade(xdata,*pfit)), label='Linear Fit p=a*m+b\na=%s mbar\nb=%s mbar'%tuple(pp))
   #plt.plot(x, y, label='noice')
   plt.legend(prop={'size':fig_legendsize})
   plt.grid()
   plt.tick_params(labelsize=fig_labelsize)
   plt.xlabel(names[1])
   plt.ylabel(names[0])
   plt.savefig("MP1/img/%s.pdf"%(fname))
   plt.show()






#end
