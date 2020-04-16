from astropy.io import fits
from pylab import *
import numpy as np
from astroML.time_series import lomb_scargle
from scipy.optimize import curve_fit
from astropy.io import ascii
import peakutils
from numpy.fft import *
from scipy.signal import argrelextrema
import matplotlib.gridspec as gridspec
from mpmath import cot,csc
from matplotlib.ticker import Locator
from copy import deepcopy
#from colours import Color
import pdb

def manage_requirements(target):

    print target

    dataloc_Joe = '/Users/lbiddle/PycharmProjects/K2/temporary/CITau_lc_shift.txt'
    dataloc_k2sc = '/Users/lbiddle/PycharmProjects/K2/oldstuff/data/' + target + '/'
    k2scfile = 'EPIC_' + EPICs[target] + '_mast.fits'
    save_here = 'CITau_AccretionSigs/SineTest/'

    epic = EPICs[target]

    return epic, dataloc_Joe, dataloc_k2sc, k2scfile, save_here

def read_k2sc(loc,file):
    #K2SC FITS EXTENSIONS
    hdulist_k2sc = fits.open(loc + file)
    header_k2sc = hdulist_k2sc[0].header
    tbldata_k2sc = hdulist_k2sc[1].data
    cols_k2sc = hdulist_k2sc[1].columns
    #CONTENTS OF FITS EXTENSIONS
    cadence_k2sc = tbldata_k2sc.field('cadence')
    qual_k2sc = tbldata_k2sc.field('quality')
    mflags = tbldata_k2sc.field('mflags')
    trtime = tbldata_k2sc.field('trtime')
    trposi = tbldata_k2sc.field('trposi')
    bjd_k2sc = tbldata_k2sc.field('time')
    bjd_k2sc = bjd_k2sc + hdulist_k2sc[1].header['BJDREFI']+hdulist_k2sc[1].header['BJDREFF']
    flux_k2sc = tbldata_k2sc.field('flux')
    err_k2sc = tbldata_k2sc.field('error')
    vflux = flux_k2sc - trposi + nanmedian(trposi)
    sflux = tbldata_k2sc.field('flux') - trtime + nanmedian(trposi)


    #pdb.set_trace()

    wherenan = isnan(vflux)
    nonan = where(wherenan == False)[0]
    bjd_k2sc = bjd_k2sc[nonan]
    sflux = sflux[nonan]
    err_k2sc = err_k2sc[nonan]
    vflux = vflux[nonan]

    #pdb.set_trace()

    return bjd_k2sc, vflux, err_k2sc, sflux, qual_k2sc, mflags, cadence_k2sc, cols_k2sc

def read_data(lcname):
    filedata = ascii.read(lcname, data_start=0)
    lc_time = array(filedata['col1'])
    lc_flux = array(filedata['col2'])
    lc_flux_err = array(filedata['col3'])

    return lc_time,lc_flux,lc_flux_err

# --------------------------------------------------------------

def make2sine(x,P1,P2):

    y = np.sin(((2.*np.pi)/P1)*x) + np.sin(((2.*np.pi)/P2)*x)

    return y

def lombscargle(x, y, yerr, Pmin=0.5, Pmax=70, res=10000):
    periods = linspace(Pmin, Pmax, res)
    ang_freqs = 2 * pi / periods
    powers = lomb_scargle(x, y, yerr, ang_freqs, generalized=True)
    return periods, powers

# --------------------------------------------------------------

def plot_sine(target, x, y, save_here):

    fig = plt.figure(num=None, figsize=(7, 4), facecolor='w', dpi=300)

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.0,right=1.,hspace=0.0)
    ax = plt.subplot(gs1[0, 0])

    font_size = 'medium'

    ax.set_ylabel('Sine1 + Sine2', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('Time (BJD - 2450000)', fontsize=font_size, style='normal', family='sans-serif')
    #ax.set_xlim([min(time1) - 1, max(time1) + 1])
    ax.plot(x, y, color='#000000',lw=0.85)
    ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)

    fig.savefig(save_here+target+'_sine.pdf', format='pdf', bbox_inches='tight')
    close()

def plot_periodogram(target, x, y, save_here):

    fig = plt.figure(num=None, figsize=(5, 5), facecolor='w', dpi=300)

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.0,right=1.,hspace=0.0)
    ax = plt.subplot(gs1[0, 0])

    font_size = 'medium'

    ax.set_ylabel('Power', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('Period', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_ylim([0,1])
    ax.plot(x, y, color='#000000',lw=0.85)

    peaks = peakutils.indexes(y, thres=0.2, min_dist=10)
    for i in peaks:
        ax.vlines(x[peaks], 0, 1, color='#000000', lw=0.5, alpha=0.2, linestyles='--')
        # ax2.text(periods1[i] - 0.58, powers1[i] + 0.015, '%5.2f' % periods1[i], horizontalalignment='center', fontsize='x-small', style='normal', family='sans-serif')
        ax.text(x[i], y[i] + 0.015, '%5.2f' % x[i], horizontalalignment='center', fontsize='small', style='normal', family='sans-serif')


    ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)
    fig.savefig(save_here+target+'_sine_periodogram.pdf', format='pdf', bbox_inches='tight')
    close()

# --------------------------------------------------------------

def sine(in_vars,a,b,c):

    x = in_vars[0]
    P = in_vars[1]

    return a*np.sin(((2.*np.pi)/P)*x + b) + c

def fit_sine(x,y,Pfit):

    popt, pcov = curve_fit(sine,[x,Pfit],y)
    sine_fit = sine([x,Pfit],*popt)

    return sine_fit, popt[-1], list(popt)

def sine2(in_vars,a,b,c,d,e,f):

    x = in_vars[0]
    P1 = in_vars[1]
    P2 = in_vars[2]

    return a*np.sin(((2.*np.pi)/P1)*x + b) + c*np.sin(((2.*np.pi)/P2)*x + d) + e

def fit_sine2(x,y,Pfit1,Pfit2):

    popt, pcov = curve_fit(sine2,[x,Pfit1,Pfit2],y)
    sine_fit2 = sine2([x,Pfit1,Pfit2],*popt)

    return sine_fit2, popt[-1]

def sine3(in_vars,a,b,c,d,e,f,g,h,i):

    x = in_vars[0]
    P1 = in_vars[1]
    P2 = in_vars[2]
    P3 = in_vars[3]

    return a*np.sin(((2. * np.pi) / P1) * x + b) + c*np.sin(((2. * np.pi) / P2) * x + d) + e*np.sin(((2. * np.pi) / P3) * x + f) + g

def fit_sine3(x,y,Pfit1,Pfit2,Pfit3):

    popt, pcov = curve_fit(sine3,[x,Pfit1,Pfit2,Pfit3],y)
    sine_fit3 = sine3([x,Pfit1,Pfit2,Pfit3],*popt)

    return sine_fit3, popt[-1]

# --------------------------------------------------------------

def plot_datafit(target, x, y, fit, fit_period, save_here):

    fig = plt.figure(num=None, figsize=(7, 4), facecolor='w', dpi=300)

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.0,right=1.,hspace=0.0)
    ax = plt.subplot(gs1[0, 0])

    font_size = 'medium'

    ax.set_title('Fit Period = %0.2f' % fit_period)
    ax.set_ylabel('Flux', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('Time (BJD - 2450000)', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlim([min(x) - 1, max(x) + 1])
    ax.plot(x, y, color='#000000',lw=1)
    ax.plot(x, fit, color='red',lw=1)
    ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)

    fig.savefig(save_here+target+'_sine_fit_%0.2f.pdf' % fit_period, format='pdf', bbox_inches='tight')
    close()

def plot_datafit2(target, x, y, fit, fit_period1, fit_period2, save_here):

    fig = plt.figure(num=None, figsize=(7, 4), facecolor='w', dpi=300)

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.0,right=1.,hspace=0.0)
    ax = plt.subplot(gs1[0, 0])

    font_size = 'medium'

    ax.set_title('Fit Periods = %0.2f,  %0.2f' % (fit_period1,fit_period2))
    ax.set_ylabel('Flux', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('Time (BJD - 2450000)', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlim([min(x) - 1, max(x) + 1])
    ax.plot(x, y, color='#000000',lw=1)
    ax.plot(x, fit, color='red',lw=1)
    ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)

    fig.savefig(save_here+target+'_sine_fit_2Period.pdf', format='pdf', bbox_inches='tight')
    close()

def plot_datafit3(target, x, y, fit, fit_period1, fit_period2, fit_period3, save_here):

    fig = plt.figure(num=None, figsize=(7, 4), facecolor='w', dpi=300)

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.0,right=1.,hspace=0.0)
    ax = plt.subplot(gs1[0, 0])

    font_size = 'medium'

    ax.set_title('Fit Periods = %0.2f,  %0.2f,  %0.2f' % (fit_period1,fit_period2,fit_period3))
    ax.set_ylabel('Flux', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('Time (BJD - 2450000)', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlim([min(x) - 1, max(x) + 1])
    ax.plot(x, y, color='#000000',lw=1)
    ax.plot(x, fit, color='red',lw=1)
    ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)

    fig.savefig(save_here+target+'_sine_fit_3Period.pdf', format='pdf', bbox_inches='tight')
    close()

# --------------------------------------------------------------

def fit_sine_movie(x,y,Pfit,index):

    popt, pcov = curve_fit(sine,[x,Pfit],y)
    popt[0] = popt[0] * (0.05*float(index))
    sine_fit = sine([x,Pfit],popt[0],popt[1],popt[2])

    return sine_fit, popt[-1], list(popt)

def plot_data_movie(target, x, y, periods, powers, fit_period, index, save_here):

    fig = plt.figure(num=None, figsize=(7, 3), facecolor='w', dpi=300)

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.0,right=1.,hspace=0.0)
    ax = plt.subplot(gs1[0, 0])

    gs2 = gridspec.GridSpec(1, 1)
    gs2.update(left=0.0, right=0.57, hspace=0.0)
    ax1 = plt.subplot(gs2[0, 0])

    gs3 = gridspec.GridSpec(1, 1)
    gs3.update(left=0.67, right=1., hspace=0.0)
    ax2 = plt.subplot(gs3[0, 0])

    font_size = 'medium'

    ax.set_title('Fit Period = %0.2f' % fit_period)
    ax1.set_ylabel('Flux', fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_xlabel('Time (BJD - 2450000)', fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_xlim([min(x) - 1, max(x) + 1])
    ax1.set_ylim([130000,260000])
    ax1.plot(x, y, color='#000000',lw=1)
    ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)

    ax2.set_ylabel('Power', fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_xlabel('Period (d)', fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_ylim([0,0.5])
    ax2.set_xlim([0, max(periods)])
    ax2.plot(periods, powers, color='#000000', lw=1)
    peaks = peakutils.indexes(powers, thres=0.2, min_dist=10)
    for j in peaks:
        ax2.vlines(periods[peaks], 0, 1, color='#000000', lw=0.5, alpha=0.2, linestyles='--')
        ax2.text(periods[j], powers[j] + 0.015, '%5.2f' % periods[j], horizontalalignment='center', fontsize='small', style='normal', family='sans-serif')
    ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)

    fig.savefig(save_here+'/Movie1/'+str(index + 1)+'_'+target+'_curve_periodogram.png', format='png', bbox_inches='tight')
    close()

def plot_datafit_movie(target, x, y, fit, fit_period, index, save_here):

    fig = plt.figure(num=None, figsize=(7, 4), facecolor='w', dpi=300)

    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.0,right=1.,hspace=0.0)
    ax = plt.subplot(gs1[0, 0])

    font_size = 'medium'

    ax.set_title('Fit Period = %0.2f' % fit_period)
    ax.set_ylabel('Flux', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('Time (BJD - 2450000)', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_ylim([140000,240000])
    #ax.plot(x, y, color='#000000',lw=1)
    ax.plot(x, fit, color='red',lw=1)
    ax.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)

    fig.savefig(save_here+'/Movie2/'+str(index + 1)+'_'+target+'_sine_fit.pdf', format='pdf', bbox_inches='tight')
    close()



EPICs = {
    'AATau' : str(247810494),
    'CITau' : str(247584113),
    'CoKuHPTauG2' : str(247592103),
    'CoKuHPTauG3' : str(247591948),
    'DFTau': str(247986526),
    'DKTau': str(248029373),
    'DMTau' : str(247047380),
    'DQTau': str(246925324),
    'FFTau': str(247592507),
    'FUOr': str(210683818),
    'FXTau': str(247805410),
    'FYTau': str(247788960),
    'GHTau': str(247763883),
    'GITau' : str(247792225),
    'GKTau': str(247791801),
    'GMTau' : str(248046139),
    'GNTau': str(247992574),
    'GVTau': str(247820507),
    'Haro6-28': str(247592919),
    'Haro6-37': str(246929818),
    'HBC422': str(247941378),
    'HBC425': str(246942563),
    'HKTau' : str(247799571),
    'HLTau' : str(210690913),
    'HQTau': str(247583818),
    'IQTau' : str(248040905),
    'IRASF04370+2559': str(248038058),
    'IRASF04385+2550': str(248018164),
    'ISTau': str(248047443),
    'ITTau': str(248055184),
    'JH92': str(247794535),
    'JH108': str(247586788),
    'LkCa15' : str(247520207),
    'TAP35': str(210662824),
    'UXTauA': str(210690735),
    'UXTauB': str(210690736),
    'UZTauE': str(248009353),
    'V1075Tau': str(210670948),
    'V1076Tau': str(247034775),
    'V710TauB': str(210699801),
    'V807Tau': str(247764745),
    'V826Tau': str(247032616),
    'V827Tau': str(210698281),
    'V830Tau': str(247822311),
    'V927Tau': str(247766994),
    'V928Tau': str(247795097),
    'V955Tau': str(247941930),
    'XZTauA': str(210690892),
    'XZTauB': str(210690886),
    'ZZTau': str(247843485),
    }



target = 'CITau'

epic, dataloc_Joe, dataloc_k2sc, k2scfile, save_here = manage_requirements(target)

time, flux, err, sflux, qual_k2sc, mflags, cadence_k2sc, cols_k2sc = read_k2sc(dataloc_k2sc, k2scfile)
time -= 2450000
#time,flux,err = read_data(dataloc_Joe)


P1 = 6.57
P2 = 9.06
P3 = 24.44
P4 = 14.27
P5 = 11.49


y = make2sine(time,P1,P2)

plot_sine(target, time, y, save_here)


periods, powers = lombscargle(time, y, np.zeros(len(time))+1e-10)

plot_periodogram(target, periods, powers, save_here)


# --------------------------------------------------------------

Pfit = P3
sine_fit, shift, popt_out = fit_sine(time,flux,Pfit)

plot_datafit(target, time, flux, sine_fit, Pfit, save_here)

# --------------------------------------------------------------

Pfit1 = P1
Pfit2 = P2
sine_fit2, shift2 = fit_sine2(time,flux,Pfit1,Pfit2)

plot_datafit2(target, time, flux, sine_fit2, Pfit1, Pfit2, save_here)

# --------------------------------------------------------------

Pfit1 = P1
Pfit2 = P2
Pfit3 = P3
sine_fit3, shift3 = fit_sine3(time,flux,Pfit1,Pfit2,Pfit3)

plot_datafit3(target, time, flux, sine_fit3, Pfit1, Pfit2, Pfit3, save_here)


plot_datafit3(target, time, flux, sine_fit3, Pfit1, Pfit2, Pfit3, save_here)

# --------------------------------------------------------------

interesting_periods = [6.57,9.06,24.44,14.34,19.21,11.55]



iterations = 20
curves = []
fits = []


Pfit = 6.57
flux_test = np.copy(flux)
for i in range(iterations):

    print i+1

    if i > 0:
        flux_test = flux - sine_fit + shift

    curves.append(flux_test)

    sine_fit, shift, popt_out = fit_sine_movie(time, flux, Pfit, i)
    fits.append(sine_fit)

    periods, powers = lombscargle(time, flux_test, np.zeros(len(time)) + 1e-10)

    plot_data_movie(target, time, flux_test, periods, powers, Pfit, i, save_here)

    plot_datafit_movie(target, time, flux_test, sine_fit, Pfit, i, save_here)


Pfit = 9.06
flux_in = np.copy(flux_test)
for k in range(iterations):

    index_addition = 20
    k = k + index_addition

    print k+1

    if k > index_addition:
        flux_test = flux_in - sine_fit + shift

    curves.append(flux_test)

    sine_fit, shift, popt_out = fit_sine_movie(time, flux_in, Pfit, k-index_addition)
    fits.append(sine_fit)

    periods, powers = lombscargle(time, flux_test, np.zeros(len(time)) + 1e-10)

    plot_data_movie(target, time, flux_test, periods, powers, Pfit, k, save_here)

    plot_datafit_movie(target, time, flux_test, sine_fit, Pfit, k, save_here)


Pfit = 14.34
flux_in = np.copy(flux_test)
for l in range(iterations):

    index_addition = 40
    l = l + index_addition

    print l+1

    if l > index_addition:
        flux_test = flux_in - sine_fit + shift

    curves.append(flux_test)

    sine_fit, shift, popt_out = fit_sine_movie(time, flux_in, Pfit, l-index_addition)
    fits.append(sine_fit)

    periods, powers = lombscargle(time, flux_test, np.zeros(len(time)) + 1e-10)

    plot_data_movie(target, time, flux_test, periods, powers, Pfit, l, save_here)

    plot_datafit_movie(target, time, flux_test, sine_fit, Pfit, l, save_here)


Pfit = 11.55
flux_in = np.copy(flux_test)
for m in range(iterations):

    index_addition = 60
    m = m + index_addition

    print m+1

    if m > index_addition:
        flux_test = flux_in - sine_fit + shift

    curves.append(flux_test)

    sine_fit, shift, popt_out = fit_sine_movie(time, flux_in, Pfit, m-index_addition)
    fits.append(sine_fit)

    periods, powers = lombscargle(time, flux_test, np.zeros(len(time)) + 1e-10)

    plot_data_movie(target, time, flux_test, periods, powers, Pfit, m, save_here)

    plot_datafit_movie(target, time, flux_test, sine_fit, Pfit, m, save_here)


Pfit = 51.09
flux_in = np.copy(flux_test)
for n in range(iterations):

    index_addition = 80
    n = n + index_addition

    print n+1

    if n > index_addition:
        flux_test = flux_in - sine_fit + shift

    curves.append(flux_test)

    sine_fit, shift, popt_out = fit_sine_movie(time, flux_in, Pfit, n-index_addition)
    fits.append(sine_fit)

    periods, powers = lombscargle(time, flux_test, np.zeros(len(time)) + 1e-10)

    plot_data_movie(target, time, flux_test, periods, powers, Pfit, n, save_here)

    plot_datafit_movie(target, time, flux_test, sine_fit, Pfit, n, save_here)


Pfit = 9.06
flux_in = np.copy(flux_test)
for o in range(iterations):

    index_addition = 20
    o = o + index_addition

    print o+1

    if o > index_addition:
        flux_test = flux_in - sine_fit + shift

    curves.append(flux_test)

    sine_fit, shift, popt_out = fit_sine_movie(time, flux_in, Pfit, o-index_addition)
    fits.append(sine_fit)

    periods, powers = lombscargle(time, flux_test, np.zeros(len(time)) + 1e-10)

    plot_data_movie(target, time, flux_test, periods, powers, Pfit, o, save_here)

    plot_datafit_movie(target, time, flux_test, sine_fit, Pfit, o, save_here)





cleaned = flux_test

Pfit1 = P1
Pfit2 = P2
sine_fit2, shift2 = fit_sine2(time,cleaned,Pfit1,Pfit2)
plot_datafit2(target, time, cleaned, sine_fit2, Pfit1, Pfit2, save_here)





