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
fig_legendsize = 15
fig_labelsize = 15
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
def custom_exp(x,A,B,C):
    return A*(np.exp(B*x)-C) # =y

def inverse_custom_exp(x,A,B,C):
    return unp.log(x/A+C)/B
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

def out_si(fn,s,u=""):
    file = open(fn,"w")
    file.write(("\\SI{%s}{%s}"%(("%s"%(s)).replace("/",""),u)))
    print(fn,": ", "%s"%(s))
    file.close()

def out(fn,s):
    file = open(fn,"w")
    file.write(("%s"%(s)).replace("/",""))
    print(fn,": ", "%s"%(s))
    file.close()

def out_si_tab(fn, tab):
    file = open(fn,"w")
    for i in range(len(tab)):
        for j in range(len(tab[i])):
            if(j!=0):
                file.write("&")
            file.write("\\SI{%s}{}"%(("%s"%(tab[i][j])).replace("/","")))
        file.write("\\\\\n")
    file.close()
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


# %% diode al

#data = np.loadtxt("MW/raw/diode-leistung.txt", skiprows = 1)
data = np.loadtxt("MW/raw/kennlinie-aletal.txt", skiprows = 1)
fig = plt.figure(figsize=fig_size)
xd = data[:,0]
yd = data[:,1]
plt.plot(xd,yd,"x",label="Messung")

fit = fit_curvefit2(xd[0:-3],yd[0:-3],custom_exp)
xfit = np.linspace(xd[0],xd[-1],4000)
yfit = custom_exp(xfit, *unv(fit))
plt.plot(unv(xfit),unv(yfit),linewidth=2, label="Exp Fit y=A*(exp(B*x)-C)\nA=%s\nB=%s\nC=%s"%(fit[0],fit[1],fit[2]))

plt.grid()
plt.legend(prop={'size':fig_legendsize})
plt.ylabel("Spannung in V")
plt.xlabel("Leistung in dBm")
plt.tick_params(labelsize=fig_labelsize)
plt.savefig("MW/img/diode-aletal.pdf")
plt.show()

data = np.loadtxt("MW/raw/diode-leistung.txt", skiprows = 1)
fig = plt.figure(figsize=fig_size)
xd = data[:,0]
yd = data[:,1]
plt.plot(xd,yd,"x",label="Messung")
fit = fit_curvefit2(xd[0:-3],yd[0:-3],custom_exp)
xfit = np.linspace(xd[0],xd[-1],4000)
yfit = custom_exp(xfit, *unv(fit))
plt.plot(unv(xfit),unv(yfit),linewidth=2, label="Exp Fit y=A*(exp(B*x)-C)\nA=%s\nB=%s\nC=%s"%(fit[0],fit[1],fit[2]))

plt.grid()
plt.legend(prop={'size':fig_legendsize})
plt.ylabel("Spannung in V")
plt.xlabel("Leistung in dBm")
plt.tick_params(labelsize=fig_labelsize)
plt.savefig("MW/img/diode-kali.pdf")
plt.show()

# %% isolator plot
names = ["MW/raw/isolator-durchlass-frequenz.txt","MW/raw/isolator-antidurchlass-frequenz.txt"]
fig = plt.figure(figsize=fig_size)
for i in range(len(names)):
    name = names[i]
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]
    data = np.loadtxt(name, skiprows = 1)
    xd = data[:,0]
    if i == 0:
        yd = inverse_custom_exp(data[:,1],*fit)
        ll = "Durchlass"
    if i == 1:
        yd = inverse_custom_exp(data[:,1],*fit)
        ll = "Sperre"
    #plt.plot(xd,yd,"x",label="Messung")
    plt.errorbar(xd,unv(yd),usd(yd),fmt="x",capsize=5,label=ll)
plt.grid()
plt.legend(prop={'size':fig_legendsize})
plt.ylabel("Leistung in dBm")
plt.xlabel("Frequenz in GHz")
plt.tick_params(labelsize=fig_labelsize)
plt.savefig("MW/img/" + "isolator" + ".pdf")
plt.show()
# %% richtkop plot
#names = glob.glob("MW/raw/*.txt")
names = ["MW/raw/richtkop-1zu2-freq.txt","MW/raw/richtkop-1zu4-freq.txt","MW/raw/richtkop-2zu4-freq.txt"]
fig = plt.figure(figsize=fig_size)
for i in range(len(names)):
    name = names[i]
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]
    data = np.loadtxt(name, skiprows = 1)
    xd = data[:,0]

    if i == 0:
        yd = 10-inverse_custom_exp(data[:,1],*fit)
        ll = "Einfügedämpfung"
    if i == 1:
        yd = 10-inverse_custom_exp(data[:,1],*fit)
        ll = "Koppeldämpfung"
    if i == 2:
        yd = 10-inverse_custom_exp(data[:,1],*fit)
        ll = "Isolation"


    #plt.plot(xd,yd,"x",label="Messung")
    plt.errorbar(xd,unv(yd),usd(yd),fmt="x",capsize=5,label=ll)
plt.grid()
plt.legend(prop={'size':fig_legendsize})
plt.ylabel("Dämpfung in dB")
plt.xlabel("Frequenz in GHz")
plt.tick_params(labelsize=fig_labelsize)
plt.savefig("MW/img/" + "richtkop" + ".pdf")
plt.show()

# %% zirkulator
#names = glob.glob("MW/raw/*.txt")
names = ["MW/raw/zirk-9594-freq-pfeil.txt","MW/raw/zirk-9594-freq-antipfeil.txt","MW/raw/zirk-12893-pfeil-freq.txt","MW/raw/zirk-12893-antipfeil-freq.txt"]
fig = plt.figure(figsize=fig_size)
for i in range(len(names)):
    name = names[i]
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]
    data = np.loadtxt(name, skiprows = 1)
    xd = data[:,0]

    yd = 10-inverse_custom_exp(data[:,1],*fit)
    if i == 0:
        ll = "Durchlassdämpfung 9594"
    if i == 1:
        ll = "Sperrdämpfung 9594"
    if i == 2:
        ll = "Durchlassdämpfung 12893"
    if i == 3:
        ll = "Sperrdämpfung 12893"
    #plt.plot(xd,yd,"x",label="Messung")
    plt.errorbar(xd,unv(yd),usd(yd),fmt="x",capsize=5,label=ll)
plt.grid()
plt.legend(prop={'size':fig_legendsize})
plt.ylabel("Dämpfung in dB")
plt.xlabel("Frequenz in GHz")
plt.tick_params(labelsize=fig_labelsize)
plt.savefig("MW/img/" + "zirkulator" + ".pdf")
plt.show()

# %% stehende welle
names = ["MW/raw/steh-100+-1-short-9594-freq.txt","MW/raw/steh-122+-1-short-9594-freq.txt"]
for i in range(len(names)):
    fig = plt.figure(figsize=fig_size)
    name = names[i]
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]
    data = np.loadtxt(name, skiprows = 1)
    xd = data[:,0]

    yd = inverse_custom_exp(data[:,1],*fit)
    xs = []
    yrr = []

    yl = 1
    for i in range(len(yd)):
        ys = yd[i]
        if ys < yl:
            yrr.append((xd[1]-xd[0])/(2*np.sqrt(3)))
            xs.append(xd[i])
    if i == 0:
        ll = "100"
    if i == 1:
        ll = "122"
    print(yrr)
    #plt.plot(xd,yd,"x",label="Messung")
    plt.errorbar(xd,unv(yd),usd(yd),fmt="x",capsize=5,label=ll)

    for x in xs:
        fig.gca().axvline(x=x,ymin=0,ymax=7, color='r')
    plt.grid()
    plt.legend(prop={'size':fig_legendsize})
    plt.ylabel("Leistung in dBm")
    plt.xlabel("Frequenz in GHz")
    plt.tick_params(labelsize=fig_labelsize)
    plt.savefig("MW/img/" + ll +  ".pdf")
    plt.show()

    xd= np.linspace(0,len(xs)-1,len(xs))

    ifit = fit_curvefit2(xd,xs,gerade,yerr=yrr)
    xfit = np.linspace(xd[0],xd[-1],4000)
    yfit = gerade(xfit, *unv(ifit))

    fig=plt.figure(figsize=fig_size)
    plt.errorbar(xd,xs,yerr=yrr,fmt='x',capsize=5,label="Fit Peaks",color = 'r')
    plt.plot(unv(xfit), unv(yfit), color = 'green',linewidth=2, label='Linear Fit f=ax+b\na=%smm,\nb=%s'%(ifit[0],ifit[1]))

    plt.grid()
    plt.legend(prop={'size':fig_legendsize})
    plt.ylabel('Frequenz in GHz')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Nummer')
    plt.savefig("MW/img/%s"%(ll + "_fit.png"))
    plt.show()
