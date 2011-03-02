import numpy
import scipy

def cttReddening(j, dj, h, dh, k, dk, **kwargs):
    xJ = 1.235
    xH = 1.662
    xK = 2.159

    if ( 'beta' in kwargs):
        beta = kwargs['beta']
    else:
        beta = -1.96

    # Slope of the T-Tauri Locus
    Mctts = 0.63046

    # Slope of the reddening vector
    Mred = (1.0-(xH/xJ)**beta)/((xH/xJ)**beta-(xK/xJ)**beta)
    
    # Computes NIR colors
    JH = j-h
    HK = h-k

    # Computes distance along reddening vector to the T-Tauri locus
    y1 = Mctts*HK + 0.4968
    x1 = (JH - y1)/(Mred - Mctts)
    denom = ( (xH/xJ)**beta - (xK/xJ)**beta)
    Aj = x1/denom
    dAj = numpy.sqrt((dj/(denom*(Mred-Mctts)))**2.0+(dh*(1.0+Mctts)/(denom*(Mred-Mctts)))**2.0+(dk*(Mctts/(denom*(Mred-Mctts))))**2.0)

    return (Aj, dAj)


def spectralReddening(wl, flux, dFlux, spt, **kwargs):
    xJ = 1.235

    if ( 'beta' in kwargs):
        beta = kwargs['beta']
    else:
        beta = -1.96
    
    wlStart = 1.1   # microns
    wlStop = 1.3    # microns

    strongLines = [1.1789, 1.1843, 1.1896, 1.1995, 1.282]
    lineWidths = [0.002, 0.002, 0.002, 0.003, 0.005]

    bm = scipy.where( (wl > wlStart) & (wl < wlStop) )[0]
    for line, width in zip(strongLines, lineWidths):
        new_bm = scipy.where( abs(wl[bm]-line) > width)
        bm = bm[new_bm[0]]

    x = wl[bm]
    y = flux[bm]
    dy = dFlux[bm]

