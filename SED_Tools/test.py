import pyfits
import numpy
import spectralSlope
#import Gnuplot
import matplotlib.pyplot as plt
import re
import glob

#plt = Gnuplot.Gnuplot()

datadir = '/home/deen/Data/StarFormation/TWA/Covey10/CoveyB59SpeX/'

infile = 'covey_sources.txt'

name = []
twa_spt = []
spt_diff = []
twa_aj = []
twa_beta = []

red_beta = -1.95

strongLines = [1.1789, 1.1843, 1.1896, 1.1995, 1.282]
lineWidths = [0.002, 0.002, 0.002, 0.003, 0.005]

df = open(datadir+infile, 'r').read().split('\n')
for line in df:
    if (len(line) > 0):
        if line[0] != ';':
            l = line.split()
            name.append(l[0])
            twa_spt.append(float(l[2]))
            spt_diff.append(float(l[1]) - float(l[2]))

for d in name:
    hdulist = pyfits.open(datadir+d, ignore_missing_end=True)
    hdr = hdulist[0].header
    dat = hdulist[0].data
    
    aj = float(hdr['AJ_SP'])
    twa_aj.append(aj)

    wl = dat[0]
    flux = dat[1]
    df = dat[2]

    print aj, d
    if (aj > 0):
        reddening = 10**(aj/2.5*(wl/1.235)**red_beta)
        flux *= reddening
    
    twa_beta.append(spectralSlope.spectralSlope(wl, flux, df, 1.1, 1.3, -0.5, strongLines = strongLines, lineWidths = lineWidths))

stddir = '/home/deen/Data/StarFormation/Standards/IRTF_standards/reduced/'

datafiles = glob.glob(stddir+'*.fits')

dwarf_spt = []
dwarf_beta = []
giant_spt = []
giant_beta = []

for d in datafiles:
    hdulist = pyfits.open(d, ignore_missing_end=True)
    hdr = hdulist[0].header
    dat = hdulist[0].data

    gt = not( re.search('III', hdr['LUM']) is None)
    dw = not( re.search('V', hdr['LUM']) is None)
    sg = not( re.search('IV', hdr['LUM']) is None)

    if ( gt | ( dw & (not sg))):
       wl = dat[0]
       flux = dat[1]
       df = dat[2]
       
       beta = spectralSlope.spectralSlope(wl, flux, df, 1.1, 1.3, -0.5, strongLines = strongLines, lineWidths = lineWidths)

       if (gt):
           giant_spt.append(float(hdr['SPTNUM']))
           giant_beta.append(beta)
       else:
           dwarf_spt.append(float(hdr['SPTNUM']))
           dwarf_beta.append(beta)

fig_width_pt = 246.0
fig_width = 7.0  # inches
inches_per_pt = 1.0/72.27
pts_per_inch = 72.27
golden_mean = (numpy.sqrt(5)-1.0)/2.0
fig_height = fig_width*golden_mean
fig_size = [fig_width, fig_height]
params = {'backend' : 'ps',
          'axes.labelsize' : 12,
          'text.fontsize' : 12,
          'legend.fontsize' : 12,
          'xtick.labelsize' : 10,
          'ytick.labelsize' : 10,
          'text.usetex' : True,
          'figure.figsize' : fig_size}
          
plt.rcParams.update(params)
fig = plt.figure(0)
plt.clf()

plt.scatter(dwarf_spt, dwarf_beta, c= 'r' )
plt.scatter(giant_spt, giant_beta, c = 'b' )
plt.scatter(twa_spt, twa_beta, c = 'g' )
plt.xlim([54, 66])
plt.ylim([-2.0, 1.0])
plt.title('TW Hydra Spectral Typing Comparison')
plt.xticks((55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65), ('K5', 'K6', 'K7', 'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6','M7'))
plt.xlabel(r'Infrared Types (This Paper)')
plt.ylabel(r'J-band Spectral Slope')
#pyplot.show()
plt.savefig('TWA_beta_slope.eps')

fig = plt.figure(0)
plt.clf()

plt.scatter(spt_diff, twa_aj, c = 'b')
plt.title('Extinction vs. Spectral Type Differences')
plt.xlabel(r'Spectral Type Difference')
plt.ylabel(r'Extinction')
plt.savefig('ajVspt.eps')

'''
twa_plot = Gnuplot.Data(twa_spt, twa_beta)
dwarf_plot = Gnuplot.Data(dwarf_spt, dwarf_beta)
giant_plot = Gnuplot.Data(giant_spt, giant_beta)
plt.plot(twa_plot, dwarf_plot, giant_plot)
'''
