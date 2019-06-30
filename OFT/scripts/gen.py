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
fig_legendsize = 18
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
resa = []
git = []
for name in names:
    data = np.loadtxt(name, skiprows = 4, usecols=(0,1), delimiter = ";")
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]
    print(nnname)

    xs = []
    before = 0
    after = 0
    center = 0
    fig=plt.figure(figsize=fig_size)
    plt.plot(data[:,0]*1000, data[:,1], '-')


    if nnname=="2_gitter_g1":
        git.append(1)
        xs = [3,6.25,9,12,15,17.5,20.5,23.5,26.25]
        center = 4
    if nnname=="2_gitter_g2":
        git.append(2)
        xs = [21,24.75,29.5,33.5,37.5,42,46,50,54,58,62,65.5,70,74.5,79,83,87,91.5,95,99]
        before = 4
        after = 0
        center = 6
    if nnname=="2_gitter_g3":
        git.append(3)
        xs = [4.5,10,15,20,26,32,38,44,50,56,62.5,68,74,79,85,91,97.5,102]
        before = 0
        after = 0
        center = 8
    if nnname=="2_gitter_g4":
        git.append(4)
        xs = [3,9,15,20,25.5,30.5,37,42,48,54.5,60,66,71,77.5,83,90,94.5,100]
        before = 0
        after = 0
        center = 9
    if nnname=="2_gitter_g5":
        git.append(5)
        xs = [2.5,9,16,23,30.5,37.5,45,52,60,67.5,75,82.5,90,98,105]
        center = 7
    for x in xs:
        fig.gca().axvline(x=x,ymin=0,ymax=1, color='r')

    xd = np.linspace(-center,len(xs)-center-1,len(xs))
    yrr = []
    for k in range(len(xs)):
        yrr.append(1)

    fit = fit_curvefit2(xd,xs,gerade,yerr=yrr)
    xfit = np.linspace(xd[0]-before,xd[-1]+after,4000)
    yfit = gerade(xfit, *unv(fit))

    resa.append(fit[0])
    bx =[]
    by =[]
    for i in range(before):
        by.append(gerade(xfit[0]+i,*unv(fit)))
        bx.append(xfit[0]+i)

    for ky in by:
        fig.gca().axvline(x=ky,ymin=0,ymax=1, color='m')

    plt.gca().set_yscale('log');
    #plt.gca().set_xscale('log');
    #plt.legend(prop={'size':fig_legendsize})
    plt.grid()
    plt.tight_layout()
    plt.ylabel('Intensität in a.u.')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Position in mm')
    plt.savefig("OFT/img/2/%s"%(nnname + ".png"))
    plt.show()

    fig=plt.figure(figsize=fig_size)

    plt.errorbar(xd,xs,yerr=yrr, fmt='x',capsize=5, label="Fit Peaks",color='r')
    plt.plot(unv(xfit), unv(yfit), color = 'green',linewidth=2, label='Linear Fit f=ax+b\na=%smm,\nb=%s'%(fit[0],fit[1]))


    if len(bx)>0:
        plt.plot(bx,by,'x',
            label="Interpolation",color='m')

    plt.grid()
    plt.legend(prop={'size':fig_legendsize})
    plt.tight_layout()
    plt.ylabel('Position u(k) in mm')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Peakordnung k')
    plt.savefig("OFT/img/2/%s"%(nnname + "_fit.png"))
    plt.show()

# %% Table
l = 632.8e-9
d = 2.715
print(resa)
uresa = unp.uarray(unv(resa),usd(resa))
out_si_tab("OFT/res/tb_2_beug", np.transpose([git,resa,d*l/(uresa/1000)*1000]))

# %% fit pyplot
names = glob.glob("OFT/data/3/*.txt")
resa2 = []
git = []
for name in names:
    data = np.loadtxt(name, skiprows = 4, usecols=(0,1), delimiter = ";")
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]
    print(nnname)

    xs = []
    before = 0
    after = 0
    center = 0
    fig=plt.figure(figsize=fig_size)
    plt.plot(data[:,0]*1000, data[:,1], '-')


    if nnname=="3_trafo_g1":
        git.append(1)
        xs = [6.36,6.8,7.35,7.99,8.3,8.7,9.25,9.67,10.24,10.77,11.25,12,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,18.7,19.3, 19.79,20.4]
        before = 2
        after = 4
        center = 18
    if nnname=="3_trafo_g2":
        git.append(2)
        #xs = [5.75,7,8.75,10.75,12.25]#,14.75,17,19,20,22,23.5,25]
        xs = [5.75,6.6,7.3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,23.4,24,24.85,25.6,26.33,27,27.9,28.7,29.5,30.3,31.0]
        before = 4
        after = 2
        center = 12
    if nnname=="3_trafo_g3":
        git.append(3)
        xs = [9.5,10.5,11.5,12.5,13.75,14.75,16,17,-1,-1,-1,-1,-1,-1,-1,26,27,28.2,29.3]
        before = 3
        after = 3
        center = 11
    if nnname=="3_trafo_g4":
        git.append(4)
        xs = [7.75,8.8,10,10.80,12,13.25,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,27.25,28.45,29.35]
        before = 4
        after = 4
        center = 11
    if nnname=="3_trafo_g5":
        git.append(5)
        xs = [2.5,4,5.25,6.5,8,9.5,10.75,12.25,13.75,-1,-1,-1,-1,-1,-1,-1,-1,26.25,27.75,29,30.5,31.75,33.25]
        after =4
        center = 12
    for x in xs:
        if x!=-1:
            fig.gca().axvline(x=x,ymin=0,ymax=1, color='r')

    xd = np.linspace(-center,len(xs)-center-1,len(xs))
    yrr = []
    rmr = []
    for k in range(len(xs)):
        if(xs[k]==-1):
            rmr.append(k)
        else:
            yrr.append(1)

    xs = np.delete(xs,rmr)
    xd = np.delete(xd,rmr)

    fit = fit_curvefit2(xd,xs,gerade,yerr=yrr)
    xfit = np.linspace(xd[0]-before,xd[-1]+after,4000)
    yfit = gerade(xfit, *unv(fit))

    resa2.append(fit[0])
    bx =[]
    by =[]
    for i in range(before):
        by.append(gerade(xfit[0]+i,*unv(fit)))
        bx.append(xfit[0]+i)

    ax =[]
    ay =[]
    for i in range(after):
        ay.append(gerade(xfit[-1]-i,*unv(fit)))
        ax.append(xfit[-1]-i)

    for ty in ay:
        fig.gca().axvline(x=ty,ymin=0,ymax=1, color='m')

    for ty in by:
        fig.gca().axvline(x=ty,ymin=0,ymax=1, color='m')

    plt.gca().set_yscale('log');
    #plt.gca().set_xscale('log');
    #plt.legend(prop={'size':fig_legendsize})
    plt.grid()
    plt.tight_layout()
    plt.ylabel('Intensität in a.u.')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Position in mm')
    plt.savefig("OFT/img/3/%s"%(nnname + ".png"))
    plt.show()

    fig=plt.figure(figsize=fig_size)

    plt.errorbar(xd,xs,yerr=yrr, fmt='x',capsize=5, label="Fit Peaks",color='r')
    plt.plot(unv(xfit), unv(yfit), color = 'green',linewidth=2, label='Linear Fit f=ax+b\na=%smm,\nb=%s'%(fit[0],fit[1]))


    if len(bx)>0:
        plt.plot(bx,by,'x',
            label="Interpolation",color='m')

    if len(ax)>0:
        plt.plot(ax,ay,'x',
            label="Interpolation",color='m')

    plt.grid()
    plt.legend(prop={'size':fig_legendsize})
    plt.tight_layout()
    plt.ylabel('Position u(k) in mm')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Peakordnung k')
    plt.savefig("OFT/img/3/%s"%(nnname + "_fit.png"))
    plt.show()

l = 632.8e-9
d2 = 0.5
print(resa2)
uresa2 = unp.uarray(unv(resa2),usd(resa2))
out_si_tab("OFT/res/tb_3_beug", np.transpose([git,resa,d2*l/(uresa2/1000)*1000,d*l/(uresa/1000)*1000]))

# %% fit pyplot
names = glob.glob("OFT/data/4/*.txt")
resa2 = []
git = []
for name in names:
    data = np.loadtxt(name, skiprows = 4, usecols=(0,1), delimiter = ";")
    nname =os.path.basename(name)
    nnname = nname.split('.')[0]
    print(nnname)

    fig=plt.figure(figsize=fig_size)
    plt.plot(data[:,0]*1000, data[:,1], '-')

    plt.gca().set_yscale('log');
    #plt.gca().set_xscale('log');
    #plt.legend(prop={'size':fig_legendsize})
    plt.grid()
    plt.tight_layout()
    plt.ylabel('Intensität in a.u.')
    plt.tick_params(labelsize=fig_labelsize)

    plt.xlabel('Position in mm')
    plt.savefig("OFT/img/4/%s"%(nnname + ".png"))
    plt.show()

# %% calc
a1 = unc.ufloat(10.5,0.3)
a2 = unc.ufloat(19.5,0.3)
b1 = unc.ufloat(10.5,0.3)
b2 = unc.ufloat(20.5,0.3)
n1 = unc.ufloat(16,1)
n2 = unc.ufloat(12,1)

print((a1-a2)/n1 * n2 /(b1-b2))
