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
fig_legendsize = 14
fig_labelsize = 12
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

# %% Kali plot
# p = peak
data = np.loadtxt("V01/data/kali.txt", skiprows = 0, delimiter = " ")
xdat = unp.uarray(data[:,0],0)#data[:,1])
xdat = unp.uarray(data[:,2],0)#data[:,3])
ydat = unp.uarray(data[:,4],0)#data[:,5])

pdata = np.loadtxt("V01/fit/" +"NaNa" +".peaks", usecols=(2,3,4,5,6,7,8),skiprows = 1)
i1=5
i2=1
p_na_hx = unc.ufloat(pdata[i1-1][0],pdata[i1-1][-1]*2./2.4)
p_na_lx = unc.ufloat(pdata[i2-1][0],pdata[i2-1][-1]*2./2.4)
pdata = np.loadtxt("V01/fit/" +"NaGe" +".peaks", usecols=(2,3,4,5,6,7,8),skiprows = 1)
i1=1
i2=2
print(unc.ufloat(pdata[i1-1][0],pdata[i1-1][-1]*2./2.4))
print(unc.ufloat(pdata[i2-1][0],pdata[i2-1][-1]*2./2.4))
p_ge_hx =unc.ufloat(pdata[i2-1][0],pdata[i2-1][-1]*2./2.4)
p_ge_lx =unc.ufloat(pdata[i1-1][0],pdata[i1-1][-1]*2./2.4)
p_na_ly = unc.ufloat(511.0,0)
p_na_hy = unc.ufloat(1274.0,0)
ydat = np.array([p_ge_lx,p_ge_hx])
ydat2 = np.array([p_na_lx,p_na_hx])
xdat = np.array([p_na_ly,p_na_hy])

fig=plt.figure(figsize=fig_size)

plt.errorbar(unv(xdat),unv(ydat),usd(ydat),fmt=' ',capsize=5,label='Ge-Detektor')
plt.errorbar(unv(xdat),unv(ydat2),usd(ydat2),fmt=' ',capsize=5,label='Na-Detektor')


# Na
fit = fit_curvefit2(unv(xdat), unv(ydat2), gerade, yerr=usd(ydat2),p0 = [5.5,-2.5])
fit_na = fit
print(1/fit)
xfit = np.linspace(unv(xdat[0]),unv(xdat[-1]),400)
yfit = gerade(xfit, *unv(fit))
plt.plot(xfit,yfit,label='Na-Linear f=ax+b\na=%s Hz/Tcm\nb=%s Hz'%(fit[0],fit[1]))

# Ge
fit = fit_curvefit2(unv(xdat), unv(ydat), gerade, yerr=usd(ydat),p0 = [5.7,-2.5])
fit_ge = fit
print(1/fit)
xfit = np.linspace(unv(xdat[0]),unv(xdat[-1]),400)
yfit = gerade(xfit, *unv(fit))
plt.plot(xfit,yfit,label='Ge-Linear f=ax+b\na=%s Hz/Tcm\nb=%s Hz'%(fit[0],fit[1]))

plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.ylabel('Kanal')
plt.tick_params(labelsize=fig_labelsize)

plt.xlabel('Energie in keV')
plt.savefig("V01/img/kali_ge.pdf")
plt.show()

# %% weird fit
#swap

d = p_na_hy - p_na_ly
m_na = (d)/(p_na_hx-p_na_lx)
m_na_x = (d)/(p_na_hx-p_na_lx +usd(p_na_hx)+usd(p_na_lx))
m_na_n = (d)/(p_na_hx-p_na_lx-usd(p_na_hx)-usd(p_na_lx))
m_ge = (d)/(p_ge_hx-p_ge_lx)
m_ge_x = (d)/(p_ge_hx-p_ge_lx + usd(p_ge_hx) + usd(p_ge_lx))
m_ge_n = (d)/(p_ge_hx-p_ge_lx - usd(p_ge_hx) - usd(p_ge_lx))
print(1/m_na)
print(1/m_ge)

kali_na = lambda x : x*m_na - m_na*p_na_lx+p_na_ly
kali_na_max = lambda x : x*m_na_x - m_na_x*(p_na_lx-usd(p_na_lx))+p_na_ly
kali_na_min = lambda x : x*m_na_n - m_na_n*(p_na_lx+usd(p_na_lx))+p_na_ly
kali_ge = lambda x : x*m_ge - m_ge*p_ge_lx+p_na_ly
kali_ge_max = lambda x : x*m_ge_x - m_ge_x*(p_ge_lx-usd(p_ge_lx))+p_na_ly
kali_ge_min = lambda x : x*m_ge_n - m_ge_n*(p_ge_lx-usd(p_ge_lx))+p_na_ly

fig=plt.figure(figsize=fig_size)

plt.errorbar(unv(ydat2),unv(xdat),xerr=usd(ydat2),fmt=' ',capsize=5,label='Na-Detektor')
plt.errorbar(unv(ydat),unv(xdat),xerr=usd(ydat),fmt=' ',capsize=5,label='Ge-Detektor')
xfit = np.linspace(unv(ydat[0])-1000,unv(ydat2[-1])+1000,400)
#Na
plt.plot(xfit,unv(kali_na_max(xfit)),label='Na-Linear m=%s b=%s'%(m_na_x, kali_na_max(0)))
plt.plot(xfit,unv(kali_na(xfit)),label='Na-Linear m=%s b=%s'%(m_na,kali_na(0)))
plt.plot(xfit,unv(kali_na_min(xfit)),label='Na-Linear m=%s b=%s'%(m_na_n, kali_na_min(0)))
m_na = mean([m_na,m_na_x,m_na_n])
b_na = mean([kali_na_max(0),kali_na(0),kali_na_min(0),])
print("mean m_na=",m_na)
print("mean b_na=",b_na)
kali_na = lambda x : x*m_na+b_na
#Ge
plt.plot(xfit,unv(kali_ge_max(xfit)),label='Ge-Linear m=%s b=%s'%(m_ge_x, kali_ge_max(0)))
plt.plot(xfit,unv(kali_ge(xfit)),label='Ge-Linear m=%s b=%s'%(m_ge,kali_ge(0)))
plt.plot(xfit,unv(kali_ge_min(xfit)),label='Ge-Linear m=%s b=%s'%(m_ge_n, kali_ge_min(0)))
m_ge = mean([m_ge,m_ge_x,m_ge_n])
b_ge=mean([kali_ge_max(0),kali_ge(0),kali_ge_min(0)])
print("mean m_ge=",m_ge)
print("mean b_ge=",b_ge)
kali_ge = lambda x : x*m_ge +b_ge


#plt.gca().set_yscale('log');
#plt.gca().set_xscale('log');
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.ylabel('Energie in keV')
plt.tick_params(labelsize=fig_labelsize)

plt.xlabel('Kanal')
plt.savefig("V01/img/kali_mix.pdf")
plt.show()
# %% import der messwerte

col = []

names = glob.glob("V01/fit/*.dat")
iter= -1
for name in names:
    iter +=1
    peakids = [1,2,-1]
    data = np.loadtxt(name, skiprows = 0, delimiter = " ")

    nname =os.path.basename(name)
    nnname = nname.split('.')[0]
    print(nname)
    pdata = np.loadtxt("V01/fit/" +nnname +".peaks", usecols=(2,3,4,5,6,7,8),skiprows = 1)

    if(nnname=="NaNa" or nnname=="NaNaCh"):
        peakids[0] = 1
        peakids[1] = 5
    if(nnname=="CsNa"):
        peakids[1] = -1
    if(nnname=="MixNa"):
        peakids = [1,3,4,2]
    if(nnname=="MixGe"):
        peakids = [1,2,7,6,9]
    if(nnname=="CsGe"):
        peakids[1] = -1
    if(nnname.startswith("Mix")):
        print(pdata[:,-1])
    if(nnname.endswith("Na")):
        xdata = kali_na(unp.uarray(data[:,0],unc_n))
        ppdata = kali_na(pdata)
        pppdata = pdata * m_na
    elif(nnname.endswith("Ge")):
        xdata = kali_ge(unp.uarray(data[:,0],unc_n))
        ppdata = kali_ge(pdata)
        pppdata = pdata * m_ge
    else:
        xdata = unp.uarray(data[:,0],unc_n)
    ydata = unp.uarray(data[:,1],np.sqrt(data[:,1]))
    model = unp.uarray(data[:,-2],unc_p)
    residual = unp.uarray(data[:,-1],unc_p)

    ybackground = model

    limit = 0.4
    peaks =[]
    xpeaks = []
    ypeaks =[]
    for i in range(len(peakids)):
        if(peakids[i] != -1):
            peaks.insert(i,unp.uarray(data[:,peakids[i]+1],unc_p))
            ybackground = ybackground -peaks[i]
            xpeak, ypeak = zip(*((x, y) for x, y in zip(xdata, peaks[i]) if y > limit))
            xpeaks.insert(i,xpeak)
            ypeaks.insert(i,ypeak)

    xmodel, ymodel = zip(*((x, y) for x, y in zip(xdata, model) if y > limit))
    xback, yback = zip(*((x, y) for x, y in zip(xdata, ybackground) if y > limit))


    #fig=plt.figure(figsize=(15,8))
    fig=plt.figure(figsize=(15,8))

    ## Plot
    if(nnname.startswith("Mix")):
        frame1=fig.add_axes((.1,.3,.8,.4))
    elif(nnname.endswith("Ch")):
        frame1=fig.add_axes((.1,.3,.8,.6))
    else:
        frame1=fig.add_axes((.1,.3,.8,.5))
    #plt.errorbar(unv(xdata),unv(ydata), usd(ydata), usd(xdata),fmt=' ', capsize=5,linewidth=2,label='Druck')
    plt.plot(unv(xdata), unv(ydata), '.',label='Messung',marker=".",markersize="2")
    plt.plot(unv(xback), unv(yback), label='Untergrund',linewidth='1')
    plt.plot(unv(xmodel), unv(ymodel), label='Modell',linewidth='1')

    for i in range(len(peakids)):
        if(peakids[i] != -1):
            if(nnname.endswith("NaCh")):
                if(i==0):
                    plt.plot(unv(xpeaks[i]), unv(ypeaks[i]), label='Peak %s'%(p_na_lx),linewidth='1')
                else:
                    plt.plot(unv(xpeaks[i]), unv(ypeaks[i]), label='Peak %s'%(p_na_hx),linewidth='1')
            elif(nnname.endswith("GeCh")):
                if(i==0):
                    plt.plot(unv(xpeaks[i]), unv(ypeaks[i]), label='Peak %s'%(p_ge_lx),linewidth='1')
                else:
                    plt.plot(unv(xpeaks[i]), unv(ypeaks[i]), label='Peak %s'%(p_ge_hx),linewidth='1')
            else:
                print("mmm",xpeaks[i][np.argmax(ypeaks[i])])
                print("ooo",ppdata[peakids[i]-1][-2])
                print("ddd",pppdata[peakids[i]-1][-1]*2/2.4)
                pos = unc.ufloat(unv(ppdata[peakids[i]-1][-2]), unp.sqrt(usd(ppdata[peakids[i]-1][-2])**2+unv(pppdata[peakids[i]-1][-1]*2/2.4)**2))
                err = unp.sqrt(usd(ppdata[peakids[i]-1][-2])**2+(pppdata[peakids[i]-1][-1]*2/2.4)**2)
                sigma = unp.sqrt((pppdata[peakids[i]-1][-1]*2/2.4)**2)
                area = unp.sqrt((pppdata[peakids[i]-1][-5])**2)
                print("lel3",area)
                col.append((nnname,pos,err,sigma,area))
                plt.plot(unv(xpeaks[i]), unv(ypeaks[i]), label='Peak %s keV'%(pos),linewidth='1')


    m = 511
    comp = lambda E : E*(1-1/(1+2*E/m))
    back = lambda E : E*(1/(1+2*E/m))
    lit = []
    if(nnname.startswith("Na") and not nnname.endswith("Ch")):
        lit=[1274,511]
    if(nnname.startswith("Co")):
        lit=[1173,1332]
    if(nnname.startswith("Cs")):
        lit=[662]
    if(nnname.startswith("Mix")):
        lit=[60,88,662,1173,1332]

    for l in lit:
        c = comp(l)
        b = back(l)
        xx = [c, c]
        k = find_nearest_index(xdata,c)
        yy = [ydata[k]/10 , ydata[k]*10]
        plt.gca().plot(xx, unv(yy), linestyle = "-.", label="Compton %s keV"%(l))
        xx = [b, b]
        k = find_nearest_index(xdata,b)
        yy = [ydata[k]/10 , ydata[k]*10]
        plt.gca().plot(xx, unv(yy), linestyle = ":", label="Rückstreu. %s keV"%(l))

    if(not nnname.endswith("Ch")):
        c= 72
        b=88
        k = find_nearest_index(xdata,c)
        yy = [ydata[k]/10 , ydata[k]*10]
        plt.gca().plot([c,c], unv(yy), linestyle = "--", label="Blei %s keV"%(72))
        xx = [b, b]
        k = find_nearest_index(xdata,b)
        yy = [ydata[k]/10 , ydata[k]*10]
        plt.gca().plot([b,b], unv(yy), linestyle = "--", label="Blei %s keV"%(88))

    plt.gca().set_yscale('log');
    #plt.legend(prop={'size':fig_legendsize+2},bbox_to_anchor=(-0.05, 0), loc=10, borderaxespad=0.)
    plt.legend(prop={'size':fig_legendsize+2},bbox_to_anchor=(0., 1., 1, 1.5), loc=8,
           ncol=3, mode="expand", borderaxespad=0.)
    #plt.legend(prop={'size':fig_legendsize})
    plt.grid()
    plt.tick_params(labelsize=fig_labelsize+2)
    plt.ylabel('Ereignisse')
    ## Residual
    frame2=fig.add_axes((.1,.1,.8,.2))
    plt.plot(unv(xdata), unv(residual), '.',marker=".",markersize="2")

    plt.ylabel('Gewichtete Residuen')
    plt.grid()
    plt.tick_params(labelsize=fig_labelsize+2)
    #plt.tick_params(labelsize=fig_labelsize+2)

    plt.xlabel('Energie in keV')
    if(nnname.endswith("Ch")):
        plt.xlabel('Kanal')
    plt.savefig("V01/img/" + nnname + ".pdf")
    plt.show()
# %% Nichtlinearität

lit=[60,88,511,662,1173,1274,1332]
fig=plt.figure(figsize=fig_size)
for (a,b,z,q,arr) in col:
    l=find_nearest_index(lit,b)
    r=1-b/lit[l]
    #print("b ",b)
    #print("l ",lit[l])
    #print("r ", r)
    if(a.endswith("Na")):
        #plt.errorbar(unv(l),unv(r),yerr=usd(r),fmt=' ',capsize=5,label=a)
        if(a.startswith("Na")):
            plt.plot(unv(lit[l]),unv(r),'o',color='blue')
        if(a.startswith("Cs")):
            plt.plot(unv(lit[l]),unv(r),'o',color='red')
        if(a.startswith("Co")):
            plt.plot(unv(lit[l]),unv(r),'o',color='orange')
        if(a.startswith("Mix")):
            plt.plot(unv(lit[l]),unv(r),'o',color='purple')

plt.plot([],[],'o',color='blue',label="Na")
plt.plot([],[],'o',color='red',label="Cs")
plt.plot([],[],'o',color='orange',label="Co")
plt.plot([],[],'o',color='purple',label="Misch")
#plt.gca().set_yscale('log');
#plt.gca().set_xscale('log');
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.ylabel('relative Differenz $1-E_m/E_r$')
plt.tick_params(labelsize=fig_labelsize)

plt.xlabel('Energie in keV')
plt.savefig("V01/img/diff_na.pdf")
plt.show()

fig=plt.figure(figsize=fig_size)
for (a,b,z,q,arr) in col:
    l=find_nearest_index(lit,b)
    r=1-b/lit[l]
    #print("b ",b)
    #print("l ",lit[l])
    #print("r ", r)
    if(a.endswith("Ge")):
        #plt.errorbar(unv(l),unv(r),yerr=usd(r),fmt=' ',capsize=5,label=a)
        if(a.startswith("Na")):
            plt.plot(unv(lit[l]),unv(r),'o',color='blue')
        if(a.startswith("Cs")):
            plt.plot(unv(lit[l]),unv(r),'o',color='red')
        if(a.startswith("Co")):
            plt.plot(unv(lit[l]),unv(r),'o',color='orange')
        if(a.startswith("Mix")):
            plt.plot(unv(lit[l]),unv(r),'o',color='purple')

plt.plot([],[],'o',color='blue',label="Na")
plt.plot([],[],'o',color='red',label="Cs")
plt.plot([],[],'o',color='orange',label="Co")
plt.plot([],[],'o',color='purple',label="Misch")
#plt.gca().set_yscale('log');
#plt.gca().set_xscale('log');
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.ylabel('relative Differenz $1-E_m/E_r$')
plt.tick_params(labelsize=fig_labelsize)

plt.xlabel('Energie in keV')
plt.savefig("V01/img/diff_ge.pdf")
plt.show()



# %% abs Nichtlinearität in log
lit=[60,88,511,662,1173,1274,1332]
fig=plt.figure(figsize=fig_size)
for (a,b,z,q,arr) in col:
    l=find_nearest_index(lit,b)
    r=np.abs(1-b/lit[l])
    #print("b ",b)
    #print("l ",lit[l])
    #print("r ", r)
    if(a.endswith("Na")):
        #plt.errorbar(unv(l),unv(r),yerr=usd(r),fmt=' ',capsize=5,label=a)
        if(a.startswith("Na")):
            plt.errorbar(unv(lit[l]),unv(r),usd(r),fmt='o',color='blue',capsize=5)
        if(a.startswith("Cs")):
            plt.errorbar(unv(lit[l]),unv(r),usd(r),fmt='o',color='red',capsize=5)
        if(a.startswith("Co")):
            plt.errorbar(unv(lit[l]),unv(r),usd(r),fmt='o',color='orange',capsize=5)
        if(a.startswith("Mix")):
            plt.errorbar(unv(lit[l]),unv(r),usd(r),fmt='o',color='purple',capsize=5)

plt.plot([],[],'o',color='blue',label="Na")
plt.plot([],[],'o',color='red',label="Cs")
plt.plot([],[],'o',color='orange',label="Co")
plt.plot([],[],'o',color='purple',label="Misch")
plt.gca().set_yscale('log');
#plt.gca().set_xscale('log');
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.ylabel('Betrag der relative Differenz $|1-E_m/E_r|$')
plt.tick_params(labelsize=fig_labelsize)

plt.xlabel('Energie in keV')
plt.savefig("V01/img/diff_na_log.pdf")
plt.show()

fig=plt.figure(figsize=fig_size)
for (a,b,z,q,arr) in col:
    l=find_nearest_index(lit,b)
    r=np.abs(1-b/lit[l])
    #print("b ",b)
    print("l ",lit[l])
    print("r ", r)
    if(a.endswith("Ge")):
        #plt.errorbar(unv(lit[l]),unv(r),yerr=usd(r),fmt=' ',capsize=5,label=a)

        if(a.startswith("Na")):
            plt.errorbar(unv(lit[l]),unv(r),usd(r),fmt='o',color='blue',capsize=5)
        if(a.startswith("Cs")):
            plt.errorbar(unv(lit[l]),unv(r),usd(r),fmt='o',color='red',capsize=5)
        if(a.startswith("Co")):
            plt.errorbar(unv(lit[l]),unv(r),usd(r),fmt='o',color='orange',capsize=5)
        if(a.startswith("Mix")):
            plt.errorbar(unv(lit[l]),unv(r),usd(r),fmt='o',color='purple',capsize=5)

plt.plot([],[],'o',color='blue',label="Na")
plt.plot([],[],'o',color='red',label="Cs")
plt.plot([],[],'o',color='orange',label="Co")
plt.plot([],[],'o',color='purple',label="Misch")
plt.gca().set_yscale('log');
#plt.gca().set_xscale('log');
plt.legend(prop={'size':fig_legendsize})
plt.grid()
plt.ylabel('Betrag der relative Differenz $|1-E_m/E_r|$')
plt.tick_params(labelsize=fig_labelsize)

plt.xlabel('Energie in keV')
plt.savefig("V01/img/diff_ge_log.pdf")
plt.show()

# %% Energie

fig=plt.figure(figsize=fig_size)
ax1 = plt.gca()
ax2 = ax1.twinx()
for (a,b,z,q,arr) in col:
    l=find_nearest_index(lit,b)
    #r=np.abs(1-b/lit[l])
    #print("b ",b)
    #print("z ",z)
    #print("q ",q)
    #print("l ",lit[l])
    #print("r ", r)
    eq = q/b
    if(a.startswith("Na")):
        color = 'blue'
    if(a.startswith("Cs")):
        color='red'
    if(a.startswith("Co")):
        color='orange'
    if(a.startswith("Mix")):
        color='purple'
    if(a.endswith("Ge")):
        ax1.errorbar(unv(b),unv(eq),yerr=usd(eq),fmt='x',capsize=5,color=color)
    if(a.endswith("Na")):
        ax2.errorbar(unv(b),unv(eq),yerr=usd(eq),fmt='o',capsize=5,color=color)
        print("q ",  q)
        print("b ",b)
        print("eq ", eq)

ax1.errorbar([],[],[],color='blue',label="Na",capsize=5)
ax1.errorbar([],[],[],color='red',label="Cs",capsize=5)
ax1.errorbar([],[],[],color='orange',label="Co",capsize=5)
ax1.errorbar([],[],[],color='purple',label="Misch",capsize=5)

ax2.plot([],[],'x',color='black',label="Ge-Detektor")
ax2.plot([],[],'o',color='black',label="NaI-Detektor")

ax2.legend(prop={'size':fig_legendsize},loc=9)
ax1.legend(prop={'size':fig_legendsize})
ax1.grid()
ax2.grid()
ax1.set_ylabel('Ge-Detektor Energieauflösung $\\Delta E$')
ax1.tick_params(labelsize=fig_labelsize)

ax2.set_ylabel('NaI-Detektor Energieauflösung $\\Delta E$')
ax2.tick_params(labelsize=fig_labelsize)

ax1.set_xlabel('Energie in keV')
plt.savefig("V01/img/res.pdf")
plt.show()
# %% ion


fig=plt.figure(figsize=fig_size)
ax1 = plt.gca()
ax2 = ax1.twinx()
for (a,b,z,q,arr) in col:
    l=find_nearest_index(lit,b)
    #r=np.abs(1-b/lit[l])
    #print("b ",b)
    #print("z ",z)
    #print("q ",q)
    #print("l ",lit[l])
    #print("r ", r)
    eq = q**2/b
    if(a.startswith("Na")):
        color = 'blue'
    if(a.startswith("Cs")):
        color='red'
    if(a.startswith("Co")):
        color='orange'
    if(a.startswith("Mix")):
        color='purple'
    if(a.endswith("Ge")):
        ax1.errorbar(unv(b),unv(eq),yerr=usd(eq),fmt='x',capsize=5,color=color)
    if(a.endswith("Na")):
        ax2.errorbar(unv(b),unv(eq),yerr=usd(eq),fmt='o',capsize=5,color=color)
        print(q/b)

ax1.errorbar([],[],[],color='blue',label="Na",capsize=5)
ax1.errorbar([],[],[],color='red',label="Cs",capsize=5)
ax1.errorbar([],[],[],color='orange',label="Co",capsize=5)
ax1.errorbar([],[],[],color='purple',label="Misch",capsize=5)

ax2.plot([],[],'x',color='black',label="Ge-Detektor")
ax2.plot([],[],'o',color='black',label="NaI-Detektor")

ax2.legend(prop={'size':fig_legendsize},loc=9)
ax1.legend(prop={'size':fig_legendsize})
plt.grid()
ax1.set_ylabel('Ge-Detektor Mittlere Ionisationsenergie $I$ in keV')
ax1.tick_params(labelsize=fig_labelsize)

ax2.set_ylabel('NaI-Detektor Mittlere Ionisationsenergie $I$ in keV')
ax2.tick_params(labelsize=fig_labelsize)

ax1.set_xlabel('Energie in keV')
plt.savefig("V01/img/ion.pdf")
plt.show()

# %% Effizienz
lit=[60,88,511,662,1173,1274,1332]
lita= [3.35e3,1.17e4,-1,2.38e3,1.18e3,-1,1.18e3]
ref = [unc.ufloat(0.003717,0.000012),unc.ufloat(0.0083,0.0006)]
time = [1879,3791]
fig=plt.figure(figsize=fig_size)
ax1 = plt.gca()
for (a,b,z,q,arr) in col:
    l=find_nearest_index(lit,b)
    if(a.startswith("Mix")):
        if(a.endswith("Ge")):
            eq = arr/lita[l]/ref[0]/time[0]
            print(b, " - ", eq)
            color = 'red'
            ax1.errorbar(unv(b),unv(eq),yerr=usd(eq),fmt='x',capsize=5,color=color)
        if(a.endswith("Na")):
            eq = arr/lita[l]/ref[1]/time[1]
            color = 'blue'
            print(a, " ", eq)
            ax1.errorbar(unv(b),unv(eq),yerr=usd(eq),fmt='o',capsize=5,color=color)


ax1.errorbar([],[],[],[],'x',color='red',label="Ge-Detektor")
ax1.errorbar([],[],[],[],'o',color='blue',label="NaI-Detektor")

ax1.legend(prop={'size':fig_legendsize})
#ax1.legend(prop={'size':fig_legendsize})
plt.grid()
ax1.set_ylabel('Normalisierte Relative Effizienz $ε$')
ax1.tick_params(labelsize=fig_labelsize)


ax1.set_xlabel('Energie in keV')
plt.savefig("V01/img/eff.pdf")
plt.show()

# %% Erz

data = np.loadtxt("V01/fit/ErzGe.xy", skiprows = 0, delimiter = " ")
erzE = kali_ge(data[:,0])
erzY = data[:,1]
reihen = ["U-Ra", "U-Ac", "Th", "Np"]
reihe = {}
for n in reihen:
    print("Reihe: ", n)
    data = np.loadtxt("V01/lit/gammas_%s.csv" % n, skiprows = 1, delimiter=",", dtype={'names': ('Eg', 'Ig', 'decay', 'halflife', 'unit', 'parent'), 'formats': ('f4', 'f4', 'S4', 'f4', 'S2', 'S8')})
    reihe[n] = data
print(len(reihe["U-Ra"]))
found5 = {}
for n in reihen:
    found5[n] = list(filter(lambda x : x[1]>5, reihe[n]))
mask = {} # 0: no way, 1: bissl verschoben, 2: kleiner peak, 3: passt genau
mask["U-Ra"] = [0,0,0,0,0,0,0,0,0,0,3,3,3,3,3,3,0,0]
mask["U-Ac"] = [2,0,3,0,0,2,0,0,2,2,0,2,2,0,3]
mask["Th"] = [0,3,2,0,0,3,0,0,0,0,3,0,0,3,0,2,2,2,0,0]
print(found5["Th"])
mask["Np"] = [0,0,2,3,0,0,0,0,2,0,0]
percent = 5.
c = ['C3', 'C1', 'C3', 'C7']
print(c)
for r, i in zip(reihen, range(4)):
    print("Reihe: ", r, len(found5[r]), len(mask[r]))
    fig=plt.figure(figsize=fig_size)
    ax = plt.gca()

    #erzX, erzY = erz
    #erzE = energyHalb(erzX)

    for (Eg, Ig, decay, time, unit, parent), m, t in zip(found5[r], mask[r], range(len(found5[r]))):
        #if m >= 2:
            #found[r].append((Eg, Ig, decay, time, unit, parent))
            #print("found: ", Eg, Ig, parent)
        coll="blue"
        if m>=1:
            coll="orange"
        if m>=2:
            coll = "red"
        if m >=3:
            coll="black"
        if m>=2:
            plt.plot([Eg,Eg], [linestart, lineend], linewidth=1, ls="-.", color=coll,label="%s keV %s" % (Eg, str(parent)[3:-2]))
        else:
            plt.plot([Eg,Eg], [linestart, lineend], linewidth=1, ls="-.", color=coll)#,label="%s keV %s" % (Eg, str(parent)[3:-2]))
            #if Eg <= 1200:
             #   plt.annotate(xy = (Eg,1+t), s=t)
    #plot original data
    plt.plot(unv(erzE), unv(erzY), label="Messpunkte", ls=" ",marker=".",markersize="2")

    plt.grid()
    plt.xlabel("Energie in $keV$")
    plt.ylabel("Ereignisse")
    plt.xlim(unv(erzE[0]),unv(erzE[-1]))
    #plt.xlim(600, 700)
    #plt.ylim(unv(max(min(activeY),1)), unv(3*max(max(activeY),1)))
    #plt.legend(bbox_to_anchor=(0.68,0.87), loc="upper right", bbox_transform=fig.transFigure, prop={'size':fig_legendsize})

    plt.legend(prop={'size':fig_legendsize})
    plt.tick_params(labelsize=fig_labelsize)
    ax.set_yscale('log')
    plt.savefig("V01/img/erz_%s.pdf" % r)
    plt.show()

# %% Wombokombo

c = ['black', 'red', 'green', 'orange']
fig=plt.figure(figsize=fig_size)
ax = plt.gca()


patches = []
gammas = []

for r, i in zip(reihen, range(4)):
    patches.append(mpatches.Patch(color=c[i], label='%s Reihe' % r))
    for (Eg, Ig, decay, time, unit, parent),m in zip(found5[r],mask[r]):
        if m >= 3:
            #print("found: ", Eg, Ig, parent)
            gammas.append((Eg, Ig, decay, time, unit, parent))
            plt.plot([Eg,Eg], [linestart,lineend], linewidth=1, ls="-.",color=c[i])

#plot original data
plt.plot(unv(erzE), unv(erzY), label="Messpunkte", ls=" ",marker=".",markersize="3")

plt.grid()
plt.xlabel("Energie in $keV$")
plt.ylabel("Ereignisse")
plt.xlim(unv(erzE[0]),unv(erzE[-1]))
#plt.xlim(600, 700)
#plt.ylim(unv(max(min(activeY),1)), unv(3*max(max(activeY),1)))
#plt.legend(bbox_to_anchor=(0.68,0.87), loc="upper right", bbox_transform=fig.transFigure, prop={'size':fig_legendsize})
plt.legend(handles=patches, prop={'size':fig_legendsize},loc=8)
plt.tick_params(labelsize=fig_labelsize)
ax.set_yscale('log')
plt.savefig("V01/img/erz_alles.pdf")
plt.show()
#end
