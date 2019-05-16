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

def custom2(x,d,b):
    c = 299792458 # m/s
    m = 9.109e-31 # kg
    e = 1.602e-19 # C
    a = 1.0/137
    Z = 56
    p = (x/100)*e
    v = np.sqrt(p**2/(m**2+p**2/c**2))
    n=(Z*a*c/v)
    return d*p**2*(b-np.sqrt(p**2*c**2+m**2*c**4))**2 *n/(1-np.exp(-2*np.pi*n))
    #return d*p**2*(b-p*c)**2 *n/(1-np.exp(-2*np.pi*n))

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

unc_x = 0.002/math.sqrt(3)*0
unc_y = 0.005/math.sqrt(3)*0
unc_t = 0.02
typ = [ "Zeitdifferenzen","Zeitkalibrierung","Positronium_Zeitdifferenz","Energiespektrum_Start", "Energiespektrum_Stop", ]
position = 39.76
kali = 0.64/514.8
width = 0.001 # == 1Grad

data = np.genfromtxt("V06/data/Langzeitmessung.txt", skip_header=1)
xdata = unp.uarray(data[:,1],0)
ydata = unp.uarray(data[:,3],np.sqrt(data[:,3]))
tdata = unp.uarray(data[:,2],0)
print('T=%s tage'%(np.sum(tdata)/1000/60/60/24))

fig=plt.figure(figsize=fig_size)
plt.bar(unv(xdata), unv(ydata), width=width, color='r', yerr=usd(0), label= 'Messpunkte')
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.xlabel('Spannung $U$ in V')
plt.ylabel('Ereignisrate R in Hz')
plt.savefig(("V06/img/raw.pdf"))
plt.show()
ydata = ydata/(tdata/1000)
print(xdata)
#################################################################### Untergrund
fig=plt.figure(figsize=fig_size)
#plt.errorbar(unv(xdata), unv(ydata),usd(ydata), color='r', label= 'Messpunkte')
plt.plot(unv(xdata), unv(ydata), color='r', label= 'Messpunkte')
#plt.bar(unv(xdata), unv(ydata), width=width, color='r', yerr=usd(0), label= 'Messpunkte')
lb=find_nearest_index(xdata,0.25)
rb=find_nearest_index(xdata,0.0)
llb=find_nearest_index(xdata,2)
rrb=find_nearest_index(xdata,1.75)
fit = fit_curvefit2(unv(np.append(xdata[lb:rb],xdata[llb:rrb])), unv(np.append(ydata[lb:rb],ydata[llb:rrb])), lambda x,a : gerade(x,0,a), p0 = [10])

xfit = np.linspace(xdata[0],xdata[-1],4000)
print(fit)
underground=fit[0]
yfit = gerade(xfit, 0,*unv(fit))
plt.plot(unv(xfit), unv(yfit), color = 'blue',linewidth=2, label='Untergrund %s Hz'%(underground))#\n$T_0$=%s $\\mu s$\n$N$=%s\n$\Delta T$=%s $\\mu s$'%tuple(fit))

plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Spannung $U$ in V')
plt.ylabel('Ereignisrate R in Hz')
plt.savefig(("V06/img/untergrund.pdf"))
plt.show()
############################################################################Kali
wzoom = 0.95
wwzoom = 1.075
lb=find_nearest_index(xdata,1.075)
rb=find_nearest_index(xdata,0.95)
yydata = ydata[lb:rb]-underground
xxdata = xdata[lb:rb]

fig=plt.figure(figsize=fig_size)
plt.plot(unv(xxdata), unv(yydata), color='r', label= 'Messpunkte')

llb=find_nearest_index(xxdata,1.01)
rrb=find_nearest_index(xxdata,0.997)
fit = fit_curvefit2(unv(xxdata[llb:rrb]), unv(yydata[llb:rrb]), custom, p0 = [1,33,0.005])
xfit = np.linspace(unv(xxdata[rrb]),unv(xxdata[llb]),400)
print(fit)
yfit = custom(xfit, *unv(fit))
plt.plot(unv(xfit), unv(yfit), color = 'blue',linewidth=2, label='Gauss Fit\n$U_0$=%s V\n$R$=%s Hz\n$\Delta U$=%s V'%tuple(fit))

llb=find_nearest_index(xxdata,1.0425)
rrb=find_nearest_index(xxdata,1.025)
fit = fit_curvefit2(unv(xxdata[llb:rrb]), unv(yydata[llb:rrb]), custom, p0 = [1,7,0.007])
xfit = np.linspace(unv(xxdata[rrb]),unv(xxdata[llb]),400)
print(fit)
yfit = custom(xfit, *unv(fit))
plt.plot(unv(xfit), unv(yfit), color = 'green',linewidth=2, label='Gauss Fit\n$U_0$=%s V\n$R$=%s Hz\n$\Delta U$=%s V'%tuple(fit))
#plt.plot(unv(xdata), unv(ydata), color='r', label= 'Messpunkte')
#plt.bar(unv(xdata), unv(ydata), width=width, color='r', yerr=usd(0), label= 'Messpunkte')

plt.legend(prop={'size':fig_legendsize})
plt.grid()
#plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Spannung $U$ in V')
plt.ylabel('Ereignisrate R in Hz')
plt.savefig(("V06/img/kalibration.pdf"))
plt.show()
#################################################################################### Kali 2

kali = unc.ufloat(0.384,0.089)
lpeak = unc.ufloat(1.0033,0.005)
lbrho = unc.ufloat(0.33814,0.00005)
fig=plt.figure(figsize=fig_size)
#plt.errorbar(unv(xdata), unv(ydata),usd(ydata), color='r', label= 'Messpunkte')
rb=find_nearest_index(xdata,0.3125)
lb=find_nearest_index(xdata,1.1)
yydata= ydata[lb:rb]-underground
xxdata= xdata[lb:rb]*kali-lpeak*kali+lbrho
plt.plot(unv(xxdata), unv(yydata), color='r', label= 'Messpunkte')

lb=find_nearest_index(xxdata,0.11)
rb=find_nearest_index(xxdata,0.1)
llb=find_nearest_index(xxdata,0.31)
rrb=find_nearest_index(xxdata,0.3)

fit = fit_curvefit2(unv(np.append(xxdata[lb:rb],xxdata[llb:rrb])), unv(np.append(yydata[lb:rb],yydata[llb:rrb])), gerade, p0 = [15,-2])
xfit = np.linspace(xxdata[lb],xxdata[rrb],4000)
print(fit)
ofit = fit
yfit = gerade(xfit, *unv(fit))
plt.plot(unv(xfit), unv(yfit), color = 'blue',linewidth=2, label='Linear f=a+xb\na=%s Hz\nb=%s Hz/Tcm'%(fit[0],fit[1]))#\n$T_0$=%s $\\mu s$\n$N$=%s\n$\Delta T$=%s $\\mu s$'%tuple(fit))
tttt = yydata[llb:rb]-gerade(xxdata[llb:rb],*unv(fit))
zzzz = xxdata[llb:rb]- xxdata[lb]
#fit = fit_curvefit2(unv(zzzz), unv(tttt), custom2, p0 = [1,1])
xfit = np.linspace(unv(xxdata[lb]),unv(xxdata[rrb]),4000)
fit = [1e70,1.6e-19*1000*1250]
print(fit)
yfit = custom2(xfit,*unv(fit))+gerade(xfit, *unv(ofit))
print(yfit[2001]-gerade(xfit[2001],*unv(ofit)))
#fit = [5,1.6e-19*1000*1000]
yfit = custom2(xfit,*unv(fit))+gerade(xfit, *unv(ofit))
print(yfit[2001]-gerade(xfit[2001],*unv(ofit)))
plt.plot(unv(xfit), unv(yfit), color = 'green',linewidth=2, label='Linear f=a+xb\na=%s\nb=%s Hz'%(fit[0],fit[1]))#\n$T_0$=%s $\\mu s$\n$N$=%s\n$\Delta T$=%s $\\mu s$'%tuple(fit))


plt.legend(prop={'size':fig_legendsize})
plt.grid()
#plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('$B\\rho$ in Tcm')
plt.ylabel('Ereignisrate R in Hz')
plt.savefig(("V06/img/kali.pdf"))
plt.show()


#end
