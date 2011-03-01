import pyfits
import spectralSlope
import Gnuplot
import re
import glob

plt = Gnuplot.Gnuplot()

datadir = '/home/deen/Data/StarFormation/TWA/Covey10/CoveyB59SpeX/'

infile = 'covey_sources.txt'

name = []
twa_spt = []
twa_beta = []

strongLines = [1.1789, 1.1843, 1.1896, 1.1995, 1.282]
lineWidths = [0.002, 0.002, 0.002, 0.003, 0.005]

df = open(datadir+infile, 'r').read().split('\n')
for line in df:
    if (len(line) > 0):
        if line[0] != ';':
            l = line.split()
            name.append(l[0])
            twa_spt.append(float(l[1]) + 1.2)

for d in name:
    hdulist = pyfits.open(datadir+d, ignore_missing_end=True)
    hdr = hdulist[0].header
    dat = hdulist[0].data
    
    wl = dat[0]
    flux = dat[1]
    df = dat[2]
    
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

twa_plot = Gnuplot.Data(twa_spt, twa_beta)
dwarf_plot = Gnuplot.Data(dwarf_spt, dwarf_beta)
giant_plot = Gnuplot.Data(giant_spt, giant_beta)
plt.plot(twa_plot, dwarf_plot, giant_plot)

