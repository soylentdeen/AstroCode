import numpy
import scipy
import pyfits
import scipy.optimize as optimize
import Gnuplot

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
    errfunc = lambda p, x, z, dz: numpy.abs((fitfunc(p, x) - z)*dz)
    pfit = scipy.optimize.leastsq(errfunc, coeffs, args=(numpy.asarray(x, dtype=numpy.float64),
    numpy.asarray(y,dtype=numpy.float64), numpy.asarray(dy,dtype=numpy.float64)), full_output = 1)

    if ( 'plt' in kwargs ):
        original = Gnuplot.Data(x, y, with_='lines')
        guess = Gnuplot.Data(x, z, with_='lines')
        new = Gnuplot.Data(x, pfit[0][0]*(x/wlStart)**(pfit[0][1]), with_='lines')
        kwargs['plt'].plot(original, guess, new)

    return pfit[0][1]
