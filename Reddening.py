import numpy
import scipy
import SEDTools
import scipy.optimize
import Gnuplot

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

    # coefficients of best-fit line
    dwarfs = [5.13704021e-6, -8.56406004e-4, 5.20223133e-2, -1.31035703, 8.54199287]

    if ( 'beta' in kwargs):
        beta = kwargs['beta']
    else:
        beta = -1.96
    
    wlStart = 1.1   # microns
    wlStop = 1.3    # microns

    strongLines = [1.13, 1.1789, 1.1843, 1.1896, 1.1995, 1.282]
    lineWidths = [0.02, 0.002, 0.002, 0.002, 0.003, 0.005]

    A_lambda = (wl/1.235)**(beta)

    spt_beta = numpy.polyval(dwarfs, spt)

    aj_guess = 10.0
    def fitfunc(Aj):
        return flux*10.0**((Aj*A_lambda)/2.5)
    def errfunc(Aj, beta):
        new_beta = SEDTools.spectralSlope(wl, fitfunc(Aj), dFlux, wlStart, wlStop, beta, strongLines=strongLines,
        lineWidths=lineWidths)
        return abs(new_beta-beta)

    error = errfunc(aj_guess, spt_beta)
    aj_step = -2.0
    while (error > 0.01):
        aj_guess += aj_step
        new_error = errfunc(aj_guess, spt_beta)
        if (new_error > error):
            aj_step *= -0.5
        error = new_error

    if ( 'plt' in kwargs):
        plt = kwargs['plt']
        bm = scipy.where( (wl > 1.1) & (wl < 1.105) )
        norm = numpy.mean( fitfunc(aj_guess)[bm])
        a = Gnuplot.Data(wl, fitfunc(aj_guess), with_='lines')
        b = Gnuplot.Data(numpy.linspace(1.1,1.3), (norm*(numpy.linspace(1.1,1.3)/1.1)**(spt_beta)), with_='lines')
        plt('set xrange [1.1:1.3]')
        plt.plot(a, b)
        raw_input()

    return aj_guess
