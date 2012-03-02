import numpy
import scipy
import pyfits
import scipy.optimize as optimize
#import Gnuplot

'''
spectralSlope - finds the spectral slope of an input spectrum.

inputs
===========
  - wl             - wavelength array
  - flux           - flux array
  - d_flux         - array of errors on flux points
  - wl_start       - start of spectral region
  - wl_stop        - stop of spectral region
  - plt(*)         - Gnuplot object
  - strongLines(*) - List of strong line centers to remove from consideration from the spectral slope
  - lineWidths(*)  - List of widths corresponding to the strong lines

(*) Optional
output
===========
   - beta       - powelaw spectral slope
   - dbeta      - 1 sigma error bar on beta
'''

def removeContinuum(wl, flux, dFlux, wlStart, wlStop, **kwargs):
    bm = scipy.where( (wl > wlStart) & (wl < wlStop) & numpy.isfinite(flux) )[0]
    wl = wl[bm]
    flux = flux[bm]
    dFlux = dFlux[bm]
    errors = dFlux/flux

    spectral_slope = spectralSlope(wl, flux, dFlux, wlStart, wlStop, 0.0, **kwargs)

    #plt = kwargs['plt']
    #plt('set xrange[*:*]')
    
    continuum = spectral_slope[0]*(wl/wlStart)**spectral_slope[1]
    flat = flux/continuum

    #first = Gnuplot.Data(wl, flat, with_='lines')

    strong = []
    for sl in zip(kwargs["strongLines"], kwargs["lineWidths"]):
        strong.extend( scipy.where(abs(wl - sl[0]) < sl[1])[0])

    nostrong = []
    for i in range(len(wl)):
        if not(i in strong):
            nostrong.append(i)

    mn = numpy.mean(flat[nostrong])
    sig = numpy.std(flat[nostrong])

    first_pass = scipy.where( (flat > mn-0.5*sig) & (flat < mn+2*sig) )[0]

    mn = numpy.mean(flat[first_pass])
    sig = numpy.std(flat[first_pass])

    second_pass = scipy.where( (flat > mn) & (flat < mn+2*sig) )[0]
    spectral_slope = spectralSlope(wl[second_pass], flat[second_pass], dFlux[second_pass], wlStart, wlStop, 0.0,
    **kwargs)

    '''
    while ( (abs(spectral_slope[1]) > 1e-3) & (len(second_pass) > 30) ):
        continuum = spectral_slope[0]*(wl/wlStart)**spectral_slope[1]
        flat = flat/continuum
        cont = Gnuplot.Data(wl, continuum, with_='lines')
        second = Gnuplot.Data(wl, flat, with_='lines')
        pts = Gnuplot.Data(wl[second_pass], flat[second_pass])
        mn = numpy.mean(flat[second_pass])
        sig = numpy.std(flat[second_pass])
        plt.plot(first, cont, second, pts)
        second_pass = scipy.where( (flat> mn) & (flat < mn+sig) )[0]
        spectral_slope = spectralSlope(wl[second_pass], flat[second_pass], dFlux[second_pass], wlStart, wlStop, 0.0, **kwargs)
        raw_input()
    '''

    spline = scipy.interpolate.UnivariateSpline(wl[second_pass], flat[second_pass]+sig, s=100)
    #sp = Gnuplot.Data(wl, spline(wl))
    #plt.plot(first, sp)
    #raw_input()
    #continuum = (spectral_slope[0]+sig)*(wl/wlStart)**spectral_slope[1]
    flat = flat/spline(wl)
    if 'errors' in kwargs:
        return wl, flat, errors
    else:
        return wl, flat

def spectralSlope(wl, flux, dFlux, wlStart, wlStop, beta_guess, **kwargs):
    bm = scipy.where( (wl > wlStart) & (wl < wlStop) & numpy.isfinite(flux) )[0]

    if ( 'strongLines' in kwargs ):
       for line, width in zip(kwargs['strongLines'], kwargs['lineWidths']):
           new_bm = scipy.where( abs(wl[bm]-line) > width)
           bm = bm[new_bm[0]]

    x = wl[bm]
    y = flux[bm]
    dy = dFlux[bm]

    normalization = y[0]
    z = normalization*(x/wlStart)**beta_guess

    coeffs = [normalization, beta_guess]
    
    fitfunc = lambda p, x : p[0]*(x/wlStart)**(p[1])
    errfunc = lambda p, x, z, dz: numpy.abs((fitfunc(p, x) - z)/dz)
    pfit = scipy.optimize.leastsq(errfunc, coeffs, args=(numpy.asarray(x, dtype=numpy.float64),
    numpy.asarray(y,dtype=numpy.float64), numpy.asarray(dy,dtype=numpy.float64)), full_output = 1)

    if ( 'plt' in kwargs ):
        original = Gnuplot.Data(x, y, with_='lines')
        guess = Gnuplot.Data(x, z, with_='lines')
        new = Gnuplot.Data(x, pfit[0][0]*(x/wlStart)**(pfit[0][1]), with_='lines')
        kwargs['plt'].plot(original, guess, new)
        #raw_input()

    return pfit[0]
