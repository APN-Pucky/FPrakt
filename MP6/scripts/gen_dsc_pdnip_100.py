
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

data = np.loadtxt("MP6/data/DSC/100K-mt.txt", skiprows = 3)
T = unp.uarray(data[:,0], unc_T)
flow = unp.uarray(data[:,1], unc_psi)
lead_heat = (T, flow)
fig=plt.figure(figsize=fig_size) # fullscreen or sidescreen
ax = plt.gca()

xdata, ydata = lead_heat
ax.plot(unv(xdata), unv(ydata), label = "Aufheizen")
#ax.errorbar(unv(xdata), unv(ydata), usd(ydata), usd(xdata), capsize = 4, fmt = ".", label = "Aufheißen")

# heat
# fit 310-320 °C linear
start_hb = find_nearest_index(lead_heat[0],500)
end_hb = find_nearest_index(lead_heat[0],530)
start_hb2 = find_nearest_index(lead_heat[0],380)
end_hb2 = find_nearest_index(lead_heat[0],400)
xdata,ydata = lead_heat[0][start_hb:end_hb], lead_heat[1][start_hb:end_hb]
xdata = np.append(xdata,lead_heat[0][start_hb2:end_hb2])
ydata= np.append(ydata,lead_heat[1][start_hb2:end_hb2])
heat_base = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  BASE: ", heat_base)
color = next(ax._get_lines.prop_cycler)['color']
xfit = np.linspace(395, 515, 2)
yfit = gerade(xfit, *heat_base)
ax.plot(unv(xfit), unv(yfit), color = color)
# fit peak linear
start_hb = find_nearest_index(lead_heat[0],420)
end_hb = find_nearest_index(lead_heat[0],430)
xdata,ydata = lead_heat[0][start_hb:end_hb], lead_heat[1][start_hb:end_hb]
heat_peak = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  PEAK: ", heat_peak)
xfit = np.linspace(410, 430, 2)
yfit = gerade(xfit, *heat_peak)
ax.plot(unv(xfit), unv(yfit), color = color)

T_crystal = -(heat_base[1] - heat_peak[1]) / (heat_base[0] - heat_peak[0])
print("Kristallisationstemperatur: ", T_crystal)
ax.annotate('$T_e = %.2f$ °C' % unv(T_crystal),
            xy=(unv(T_crystal), unv(gerade(T_crystal,*heat_base))), xytext=(unv(T_crystal)-30, -45),
            arrowprops=dict(facecolor=color, shrink=0.05))

xdata, ydata = lead_heat
li = find_nearest_index(xdata,395)
hi = find_nearest_index(xdata,515)
full_yfit = gerade(xdata[li:hi],*heat_base)
ax.fill_between(unv(xdata[li:hi]),unv(full_yfit),unv(ydata[li:hi]),unv(ydata[li:hi])<unv(full_yfit), color = "yellow", hatch='/',edgecolor=color)
xxdata = []
for i in range(0,len(xdata)-1):
    xxdata.append(np.abs(xdata[i]-xdata[i+1]))
xxdata.append(mean(xxdata)) # last step mean approx
q=np.sum((ydata[li:hi]-full_yfit)*xxdata[li:hi])/100*60 # mJ
print ("Integral Q heat:", q)
xpp = 450
ax.annotate('$\\Delta H_e = %.2f$ mJ' % unv(q),
            xy=(xpp, unv(gerade(xpp,*heat_base))), xytext=(xpp+2, -19),
            arrowprops=dict(facecolor="yellow", shrink=0.05))



# heat
# fit linear
start_hb = find_nearest_index(lead_heat[0],280)
end_hb = find_nearest_index(lead_heat[0],295)
xdata,ydata = lead_heat[0][start_hb:end_hb], lead_heat[1][start_hb:end_hb]
heat_base1 = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  BASE: ", heat_base1)
color = next(ax._get_lines.prop_cycler)['color']
xfit = np.linspace(250, 350, 2)
yfit = gerade(xfit, *heat_base1)
ax.plot(unv(xfit), unv(yfit), color = color)


# heat
# fit linear
start_hb = find_nearest_index(lead_heat[0],340)
end_hb = find_nearest_index(lead_heat[0],360)
xdata,ydata = lead_heat[0][start_hb:end_hb], lead_heat[1][start_hb:end_hb]
heat_base2 = fit_curvefit2(unv(xdata), unv(ydata), gerade, yerr = usd(ydata), p0 = [1, 0])
print("  BASE: ", heat_base2)
color = next(ax._get_lines.prop_cycler)['color']
xfit = np.linspace(280, 370, 2)
yfit = gerade(xfit, *heat_base2)
ax.plot(unv(xfit), unv(yfit), color = color)

xdata, ydata = lead_heat
li = find_nearest_index(lead_heat[0],300)
hi = find_nearest_index(lead_heat[0],330)
full_yfit1 = gerade(xdata[li:hi],*heat_base1)
full_yfit2 = gerade(xdata[li:hi],*heat_base2)
dis = np.abs(np.abs(ydata[li:hi]-full_yfit1)-np.abs(np.abs(ydata[li:hi]-full_yfit2)))
i = np.argmin(dis)
T_schmelz = (xdata[li:hi])[i]
print("Temp Glass: " ,T_schmelz)
xdata = [T_schmelz, T_schmelz]
ydata = [-10, 10]
ax.plot(unv(xdata), ydata, color = color, linestyle = ":")
xdata, ydata = lead_heat
ax.annotate('$T_g = %.2f$ °C' % unv(T_schmelz),
            xy=(unv(T_schmelz), unv((ydata[li:hi])[i])), xytext=(unv(T_schmelz)+19, -10),
            arrowprops=dict(facecolor=color, shrink=0.05))


ax.set_xlim([240,555])
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.tick_params(labelsize=fig_labelsize)
plt.xlabel("Temperatur $T$ [$°C$]", {'fontsize':fig_legendsize+2})
plt.ylabel("Wärmefluss $\Phi$ [$mW$]", {'fontsize': fig_legendsize+2})
plt.savefig("MP6/img/Kalorimetrie_pdnip_100.pdf")
plt.show()

#end
