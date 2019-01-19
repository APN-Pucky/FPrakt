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
k_B = unc.ufloat_fromstr("1.38064852(79)e-23") # J K-1 [0]
h = unc.ufloat_fromstr("4.135667662(25)e-15") # eV s [0]
r_e = unc.ufloat_fromstr("2.8179403227(19)e-15") # m [0]
R = unc.ufloat_fromstr("8.3144598(48)") # J mol-1 K-1 [0]
K = 273.15 # kelvin
g = 9.81 # m/s^2
rad = 360 / 2 / math.pi
grad = 1/rad
# Unsicherheiten

#######################
unc_T = 0.01 / 2 / np.sqrt(3) # digital thermometer [kelvin]
unc_psi = 0.0001 / 2 / np.sqrt(3) # digital [milli watt]

data = np.loadtxt("MP6/data/DSC/blei-50k-m-heiz.txt", skiprows = 3)
T = unp.uarray(data[:,0], unc_T)
flow = unp.uarray(data[:,1], unc_psi)
lead_heat = (T, flow)

data = np.loadtxt("MP6/data/DSC/blei-50k-m-kuehl.txt", skiprows = 3)
T = unp.uarray(data[:,0], unc_T)
flow = unp.uarray(data[:,1], unc_psi)
lead_cool = (T, flow)

# Rechnung lead

# heat
# fit 310-320 °C linear
start_hb = find_nearest_index(lead_heat[0],310)
end_hb = find_nearest_index(lead_heat[0],320)
print("  BASE: ", lead_heat[0][start_hb], " to ", lead_heat[0][end_hb])
xdata,ydata = lead_heat[0][start_hb:end_hb], lead_heat[1][start_hb:end_hb]
heat_base = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  BASE: ", heat_base)

# fit 330-332 °C linear
start_hp = find_nearest_index(lead_heat[0],330)
end_hp = find_nearest_index(lead_heat[0],332)
print("  PEAK: ", lead_heat[0][start_hp], " to ", lead_heat[0][end_hp])
xdata,ydata = lead_heat[0][start_hp:end_hp], lead_heat[1][start_hp:end_hp]
heat_peak = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  PEAK: ", heat_peak)

T_schmelz = -(heat_base[1] - heat_peak[1]) / (heat_base[0] - heat_peak[0])
print("Schmelztemperatur: ", T_schmelz)
print()

# cool
# fit 320-330 °C linear
end_cb =  find_nearest_index(lead_cool[0],320)
start_cb =  find_nearest_index(lead_cool[0],330)
print("  BASE: ", lead_cool[0][end_cb], " to ", lead_cool[0][start_cb])
xdata,ydata = lead_cool[0][start_cb:end_cb], lead_cool[1][start_cb:end_cb]
cool_base = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  BASE: ", cool_base)
# fit 312.5-313.5 °C linear
end_cp = find_nearest_index(lead_cool[0],312.5)
start_cp = find_nearest_index(lead_cool[0],313.5)
print("  PEAK: ", lead_cool[0][start_cp], " to ", lead_cool[0][end_cp])
xdata,ydata = lead_cool[0][start_cp:end_cp], lead_cool[1][start_cp:end_cp]
cool_peak = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  PEAK: ", cool_peak)
T_crystal = -(cool_base[1] - cool_peak[1]) / (cool_base[0] - cool_peak[0])
print("Kristallisationstemperatur: ", T_crystal)
#cool2
# fit 320-330 °C linear
end_cb =  find_nearest_index(lead_cool[0],295)
start_cb =  find_nearest_index(lead_cool[0],300)
print("  BASE: ", lead_cool[0][end_cb], " to ", lead_cool[0][start_cb])
xdata,ydata = lead_cool[0][start_cb:end_cb], lead_cool[1][start_cb:end_cb]
cool_base2 = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  BASE: ", cool_base)
# fit 312.5-313.5 °C linear
end_cp = find_nearest_index(lead_cool[0],308)
start_cp = find_nearest_index(lead_cool[0],309.5)
print("  PEAK: ", lead_cool[0][start_cp], " to ", lead_cool[0][end_cp])
xdata,ydata = lead_cool[0][start_cp:end_cp], lead_cool[1][start_cp:end_cp]
cool_peak2 = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  PEAK: ", cool_peak)

T_crystal2 = -(cool_base2[1] - cool_peak2[1]) / (cool_base2[0] - cool_peak2[0])
print("Kristallisationstemperatur2: ", T_crystal2)
# m1 * x + b1 = m2 * x + b2
# (m1 - m2) * x = b2 - b1
# x = (b2 - b1) / (m1 - m2)
fig=plt.figure(figsize=fig_size) # fullscreen or sidescreen
ax = plt.gca()

xdata, ydata = lead_heat
ax.plot(unv(xdata), unv(ydata), label = "Aufheizen")
#ax.errorbar(unv(xdata), unv(ydata), usd(ydata), usd(xdata), capsize = 4, fmt = ".", label = "Aufheißen")
xdata, ydata = lead_cool
ax.plot(unv(xdata), unv(ydata), label = "Abkühlen")
#ax.errorbar(unv(xdata), unv(ydata), usd(ydata), usd(xdata), capsize = 4, fmt = ".", label = "Abkühlen")

color = next(ax._get_lines.prop_cycler)['color']
xdata, ydata = lead_heat

xfit = np.linspace(327, 333, 2)
yfit = gerade(xfit, *heat_peak)
ax.plot(unv(xfit), unv(yfit), color = color)

xfit = np.linspace(320, 345, 2)
yfit = gerade(xfit, *heat_base)
ax.plot(unv(xfit), unv(yfit), color = color)

li = find_nearest_index(xdata,320)
hi = find_nearest_index(xdata,340)
full_yfit = gerade(xdata[li:hi],*heat_base)
ax.fill_between(unv(xdata[li:hi]),unv(full_yfit),unv(ydata[li:hi]),unv(ydata[li:hi])>unv(full_yfit), color = "yellow", hatch='/',edgecolor=color)
xxdata = []
for i in range(0,len(xdata)-1):
    xxdata.append(np.abs(xdata[i]-xdata[i+1]))
xxdata.append(mean(xxdata)) # last step mean approx
q=np.sum((ydata[li:hi]-full_yfit)*xxdata[li:hi])/50*60 # mJ
print ("Integral Q heat:", q)
xdata = [T_schmelz, T_schmelz]
ydata = [-20, 30]
ax.plot(unv(xdata), ydata, color = color, linestyle = ":")
ax.annotate('$\\Delta H_s = %.2f$ mJ' % unv(q),
            xy=(333, unv(gerade(333,*heat_base))), xytext=(333+2, 30),
            arrowprops=dict(facecolor="yellow", shrink=0.05))
ax.annotate('$T_s = %.2f$ °C' % unv(T_schmelz),
            xy=(unv(T_schmelz), unv(gerade(T_schmelz,*heat_base))), xytext=(unv(T_schmelz)-19, 30),
            arrowprops=dict(facecolor=color, shrink=0.05))

color = next(ax._get_lines.prop_cycler)['color']

xfit = np.linspace(311.8, 315, 2)
yfit = gerade(xfit, *cool_peak)
ax.plot(unv(xfit), unv(yfit), color = color)

xfit = np.linspace(300, 320, 2)
yfit = gerade(xfit, *cool_base)
ax.plot(unv(xfit), unv(yfit), color = color)

xdata, ydata = lead_cool
hi = find_nearest_index(xdata,300)
li = find_nearest_index(xdata,320)
full_yfit = gerade(xdata[li:hi],*cool_base)
ax.fill_between(unv(xdata[li:hi]),unv(full_yfit),unv(ydata[li:hi]),unv(ydata[li:hi])<unv(full_yfit), color = "yellow", hatch='/',edgecolor=color)
xxdata = []
for i in range(0,len(xdata)-1):
    xxdata.append(np.abs(xdata[i]-xdata[i+1]))
xxdata.append(mean(xxdata)) # last step mean approx
q=np.sum((ydata[li:hi]-full_yfit)*xxdata[li:hi])/50*60
print ("Integral cool q", q)

xdata = [T_crystal, T_crystal]
ydata = [-30, 20]
ax.plot(unv(xdata), ydata, color = color, linestyle = ":")
ax.annotate('$T_e = %.2f$ °C' % unv(T_crystal),
            xy=(unv(T_crystal), unv(gerade(T_crystal,*cool_base))), xytext=(unv(T_crystal)+5, -35),
            arrowprops=dict(facecolor=color, shrink=0.05))
ax.annotate('$\\Delta H_e = %.2f$ mJ' % unv(q),
            xy=(310, unv(gerade(310,*cool_base))), xytext=(310-25, -35),
            arrowprops=dict(facecolor="yellow", shrink=0.05))
ax.set_xlim([275,350])
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel("Temperatur $T$ [$°C$]", {'fontsize':fig_legendsize+2})
plt.ylabel("Wärmefluss $\Phi$ [$mW$]", {'fontsize': fig_legendsize+2})
plt.savefig("MP6/img/Kalorimetrie_blei.pdf")
plt.show()

#end
