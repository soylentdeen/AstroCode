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
