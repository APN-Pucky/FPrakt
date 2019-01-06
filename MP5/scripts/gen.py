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

def custom(x,n):
    m = x
    l = 650.4*10**-9#unc.ufloat(630,10)*10**-9
    #l =unp.uarray([630],[10])*10**-9
    #t = unp.uarray([5],[0.1])*10**-3
    t = 5.05*10**-3#unc.ufloat(5,0.1)*10**-3
    return (n*m*l+m*m*l*l/(4*t))/(m*l+2*t*(n-1))

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
unc_x = 0.002/math.sqrt(3)
unc_y = 0.005/math.sqrt(3)
unc_w = 0.3
# import der messwerte
typ = ["industry", "selfmade"]
for t in typ:
   data = np.loadtxt("MP5/data/%s.csv"%(t), skiprows = 0, delimiter = ",")

   xdata = unp.uarray(data[:,0],unc_x)
   ydata = unp.uarray(data[:,1],unc_y)

   fig=plt.figure(figsize=fig_size)
   # Max
   print("max%s"%(np.amax(ydata)))
   #Min
   print("min%s"%(np.amin(ydata)))
   # Normalize
   ydata = normalize(ydata)*100.0
   #
   top = find_nearest_index(ydata,90.0);
   bot = find_nearest_index(ydata,10.0);
   print(ydata[top])
   print(ydata[bot])
   for i in range(top,bot):
       ydata[i] = unc.ufloat(unv(ydata[i]),usd(ydata[i])*2)
   second = False
   if(t=="industry"):
       i=31
   else:
       i=0
   plt.plot((unv(xdata[i]),unv(xdata[i])),(0,100),'k-',color='black', label="$U_{th}=%s$ V"%(xdata[i]))

       #xdata[i] = unc.ufloat(unv(xdata[i]),usd(xdata[i])*2)
   plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=2, label='Messpunkte')
   plt.plot((unv(xdata[top]),unv(xdata[top])),(0,100),'k-',color='red', label="$U_{90}=%s$ V"%(xdata[top]))
   plt.plot((unv(xdata[bot]),unv(xdata[bot])),(0,100),'k-',color='green', label="$U_{10}=%s$ V"%(xdata[bot]))
   print("dU%s"%(xdata[bot]-xdata[top]))
   #pfit, perr = fit_curvefit(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
   #pp = unp.uarray(pfit, perr)
   #xdata = np.linspace(unv(xdata[0]),unv(xdata[-1]))
   #plt.plot(xdata,unv(gerade(xdata,*pfit)), label='Linear Fit p=a*m+b\na=%s mbar\nb=%s mbar'%tuple(pp))
   #plt.plot(x, y, label='noice')
   plt.legend(prop={'size':fig_legendsize})
   plt.grid()
   plt.tick_params(labelsize=fig_labelsize)
   plt.xlabel('Angelegte Spannung $U_{LCD}$ (in V)')
   plt.ylabel('Lichttransmission (in %)')
   plt.savefig("MP5/images/%s.pdf"%(t))
   plt.show()


data = np.loadtxt("MP5/data/laser.csv", skiprows = 0, delimiter = ",")

unc_x = 0.05/math.sqrt(3)
unc_y = 3/math.sqrt(3)
xdata = unp.uarray(data[:,0],unc_x)
ydata = unp.uarray(data[:,1],unc_y)

# Normalize
#

fig=plt.figure(figsize=fig_size)
plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=1,label='Messpunkte')

#pfit, perr = fit_curvefit(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
#pp = unp.uarray(pfit, perr)
#xdata = np.linspace(unv(xdata[0]),unv(xdata[-1]))
#plt.plot(xdata,unv(gerade(xdata,*pfit)), label='Linear Fit p=a*m+b\na=%s mbar\nb=%s mbar'%tuple(pp))
#plt.plot(x, y, label='noice')
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Temperatur $T$ (in °C)')
plt.ylabel('Abstand $d$ (in mm)')
plt.savefig("MP5/images/laser.pdf")
plt.show()

#end
