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
from matplotlib.patches import Ellipse
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
fig_legendsize = 15
fig_labelsize = 15
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
def malus(x,I1,w,p,I2):
    return (I1-I2)*np.cos(w*x-p)**2+I2
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
unc_i = 0.003







# %% 4.1.1 Malus
data = data_malus = np.loadtxt("SLM/data/411.csv",skiprows = 1, delimiter=',')
ydata = unp.uarray(data[:,0],data[:,1])
xdata = unp.uarray(data[:,2],data[:,3])
fig=plt.figure(figsize=fig_size)

plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=1, label='Messpunkte')

fit = fit_curvefit2(unv(xdata), unv(ydata), malus,yerr=usd(ydata), p0 = [4.5,2*np.pi/360,0,0])

xfit = np.linspace(xdata[0],xdata[-1], 400)
xfit = xfit
yfit = malus(unv(xfit), *unv(fit))
plt.plot(unv(xfit), unv(yfit), linewidth=2, label='Malus Fit: $\\Delta I\\cdot\\cos^2(\\omega\\phi-\\theta)+I_{min}$\n$I_{max}$=%s a.u.\n$I_{min}$=%s a.u.\n$\\omega$=%s rad$^{-1}$\n$\\theta$=%s °'%(fit[0],fit[3],fit[1]*180/np.pi,fit[2]*180/np.pi+180))

out_si("SLM/res/malus_kontrast",(fit[0]-fit[3])/(fit[0]+fit[3]))



plt.legend(prop={'size':fig_legendsize},loc=4)
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Winkel $\phi$ in °')
plt.ylabel('Intensität $I$ in a.u.')
plt.savefig("SLM/img/malus.pdf")
plt.show()

# %% 4.1.2 Kontrast
unc_i = 0.003
imax = unc.ufloat(1.735,unc_i)
imin = unc.ufloat(0.064,unc_i)
out_si("SLM/res/intens_max", imax,"mW")
out_si("SLM/res/intens_min", imin, "mW")
out_si("SLM/res/intens_kontrast", (imax-imin)/(imax+imin))

# %% 4.1.3 Pixel
unc_f = 0.01/2/np.sqrt(3)
#unc_f = 0.003
unc_l = 0.001/2/np.sqrt(6)
#unc_l = 0.0002
unc_px = 10/2/np.sqrt(3)
f = unc.ufloat(0.2,unc_f) *100
bw1 = unc.ufloat(0.303,unc_l) *100
bg1 = unc.ufloat(0.003,unc_l) *100
bw2 = unc.ufloat(0.5,unc_l) *100
bg2 = unc.ufloat(0.0095,unc_l) *100
px = unc.ufloat(200,unc_px)

g1 = bg1/(bw1/f-1)
g2 = bg2/(bw2/f-1)

px1 = g1/px*1e4
px2 = g2/px*1e4
print("pixel:", px)
out_si("SLM/res/pixel1",px1,"\\mu m")
out_si("SLM/res/pixel2",px2,"\\mu m")
#'{:+.1uS}'.format(tab[i][j])
out_si_tab("SLM/res/tb_pixel",[['{:+.1uS}'.format(bw1),'{:+.1uS}'.format(bg1),'{:+.1uS}'.format(g1),px1],['{:+.1uS}'.format(bw2),'{:+.1uS}'.format(bg2),'{:+.1uS}'.format(g2),px2]])

# %% 4.2.1
f = unc.ufloat(0.2,unc_f) *100
d1 = unc.ufloat(0.025,unc_l) * 100
d2 = unc.ufloat(0.033,unc_l) *100
m1 = 3
m2 = 4
l = 632.8e-9
g1 = m1*l/(d1/(2*f)) * 1e6
g2 = m2*l/(d2/(2*f))*1e6
print(g1)
print(g2)
out_si_tab("SLM/res/tb_gitter",[
[(m1),'{:+.1uS}'.format(d1),g1],
[(m2),'{:+.1uS}'.format(d2),g2]])

# %% 4.1.4
# data
gray = [0,50,100,150,200,250]
angle = [170,169,161.5,105,98.5,94]
max = [762,757,719,722,792,795]
min = [21.83,24.7,56.6,68.9,29.3,22.7]
color = ['C0','C1','C2','C3','C4','C5']
#render
fig = plt.figure(figsize=fig_size)
for i in range(len(max)):
    mx = unc.ufloat(max[i],3)
    mn = unc.ufloat(min[i],3)
    plt.gca().add_patch(Ellipse((0,0),max[i]*2,min[i]*2,angle[i], facecolor="%s"%(gray[i]/255),edgecolor=color[i],label="Grau: %s, $\\theta$: %s°"%(gray[i],angle[i])))
    print(unp.sqrt(mx**2-mn**2))
mm = np.max(np.array([max,min]))
plt.xlim(-mm, mm)
plt.ylim(-mm, mm)
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Intensität $I$ in a.u.')
plt.ylabel('Intensität $I$ in a.u.')
plt.savefig("SLM/img/ellipse.pdf")
plt.show()


# %% 4.2.2
print(1-2/unc.ufloat(9,0.3))
print(1-4/unc.ufloat(9,0.3))
print((1-2/unc.ufloat(9,0.3))*(1-4/unc.ufloat(9,0.3)))
data = np.loadtxt("SLM/data/422_1_good.csv",skiprows = 1, delimiter=',')
xdata = unp.uarray(data[:,0],0.0)
ydata = unp.uarray(data[:,1],data[:,2]) *1e3

fig=plt.figure(figsize=fig_size)

plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=1, label='Messpunkte')

plt.gca().set_yscale('log')
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Maximum $m$')
plt.ylabel('Intensität $I$ in a.u.')
plt.savefig("SLM/img/sinc1.pdf")
plt.show()

data = np.loadtxt("SLM/data/422_2_good.csv",skiprows = 1, delimiter=',')
xdata = unp.uarray(data[:,0],0.0)
ydata = unp.uarray(data[:,1],data[:,2]) *1e3

fig=plt.figure(figsize=fig_size)

plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=1, label='Messpunkte')

plt.gca().set_yscale('log')
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Maximum $m$')
plt.ylabel('Intensität $I$ in a.u.')
plt.savefig("SLM/img/sinc2.pdf")
plt.show()

data = np.loadtxt("SLM/data/422_3_good.csv",skiprows = 1, delimiter=',')
xdata = unp.uarray(data[:,0],0.0)
ydata = unp.uarray(data[:,1],data[:,2]) *1e3

fig=plt.figure(figsize=fig_size)

plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=1, label='Messpunkte')

plt.gca().set_yscale('log')
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Maximum $m$')
plt.ylabel('Intensität $I$ in a.u.')
plt.savefig("SLM/img/sinc3.pdf")
plt.show()

# %% 4.2.3
int0 = [541,514,394,142,59,37]
int1 = [16.26,15.9,12,5.4,2.85,1.617]
gray = [255,200,150,100,50,0]

ydata0 = unp.uarray(int0,3)
ydata1 = unp.uarray(int1,0.3)

ydata = ydata1/ydata0
xdata = unp.uarray(gray,0)
out_si_tab("SLM/res/tb_beug",np.transpose([gray,ydata0,ydata1,ydata]))
fig=plt.figure(figsize=fig_size)

plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=1, label='Messpunkte')

plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Grauwert' )
plt.ylabel('Beugungswirkungsgrad $\\eta= I_{1,max}/i_{0,max}$')
plt.savefig("SLM/img/beugungsgrad.pdf")
plt.show()

# %% 4.3.1
lin = [100,75,50,25]
dis = [29.5,26.3,23.9,24.0]


xdata = unp.uarray(lin,0)
ydata = unp.uarray(dis,unc_l*1000)
print(ydata)

fig=plt.figure(figsize=fig_size)

plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=1, label='Messpunkte')

plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel('Linsenphase' )
plt.ylabel('Brennweite $f$ in cm')
plt.savefig("SLM/img/Fresnel.pdf")
plt.show()

#end
