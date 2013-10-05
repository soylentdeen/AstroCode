import numpy
import scipy
import pyfits
import scipy.optimize as optimize
import matplotlib.pyplot as pyplot

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

class InterpCubicSpline:
    """Interpolate a cubic spline through a set of points.

    Instantiate the class with two arrays of points: x and
    y = f(x).

    Inputs:

    x                 : array of x values
    y                 : array of y values (y = f(x))
    firstderiv = None : derivative of f(x) at x[0]
    lastderiv  = None : derivative of f(x) at x[-1]

    After initialisation, the instance can be called with an array of
    values xp, and will return the cubic-spline interpolated values
    yp = f(xp).

    The spline can be reset to use a new first and last derivative
    while still using the same initial points by calling the set_d2()
    method.

    If you want to calculate a new spline using a different set of x
    and y values, you'll have to instantiate a new class.

    The spline generation is loosely based on the Numerical recipes
    routines.

    Examples
    --------

    """
    def __init__(self,x,y,firstderiv=None,lastderiv=None):
        x = numpy.asarray(x).astype('float')
        lx = list(x)
        # check all x values are unique
        if any((lx.count(val) > 1) for val in set(lx)):
            raise Exception('non-unique x values were found!')
        y = numpy.asarray(y).astype('float')
        cond = x.argsort()                    # sort arrays
        self.x = x[cond]
        self.y = y[cond]
        self.npts = len(x)
        self.set_d2(firstderiv,lastderiv)

    def __call__(self,xp):
        """ Given an array of x values, returns cubic-spline
        interpolated values yp = f(xp) using the derivatives
        calculated in set_d2().
        """
        x = self.x;  y = self.y;  npts = self.npts;  d2 = self.d2

        # make xp into an array
        if not hasattr(xp,'__len__'):  xp = (xp,)
        xp = numpy.asarray(xp)

        # for each xp value, find the closest x value above and below
        i2 = numpy.searchsorted(x,xp)

        # account for xp values outside x range
        i2 = numpy.where(i2 == npts, npts-1, i2)
        i2 = numpy.where(i2 == 0, 1, i2)
        i1 = i2 - 1

        h = x[i2] - x[i1]
        a = (x[i2] - xp) / h
        b = (xp - x[i1]) / h
        temp = (a**3 - a)*d2[i1] +  (b**3 - b)*d2[i2]
        yp = a * y[i1] + b * y[i2] + temp * h*h/6.

        return yp

    def _tridiag(self,temp,d2):
        x, y, npts = self.x, self.y, self.npts
        for i in range(1,npts-1):
            ratio = (x[i]-x[i-1]) / (x[i+1]-x[i-1])
            denom = ratio * d2[i-1] + 2.       # 2 if x vals equally spaced
            d2[i] = (ratio - 1.) / denom       # -0.5 if x vals equally spaced
            temp[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
            temp[i] = (6.*temp[i]/(x[i+1]-x[i-1]) - ratio * temp[i-1]) / denom
        return temp

    def set_d2(self, firstderiv=None, lastderiv=None, verbose=False):
        """ Calculates the second derivative of a cubic spline
        function y = f(x) for each value in array x. This is called by
        __init__() when a new class instance is created.

        optional inputs:

        firstderiv = None : 1st derivative of f(x) at x[0].  If None,
                             then 2nd derivative is set to 0 ('natural').
        lastderiv  = None : 1st derivative of f(x) at x[-1].  If None,
                             then 2nd derivative is set to 0 ('natural').
        """
        if verbose:  print 'first deriv,last deriv',firstderiv,lastderiv
        x, y, npts = self.x, self.y, self.npts
        d2 = numpy.empty(npts)
        temp = numpy.empty(npts-1)

        if firstderiv is None:
            if verbose:  print "Lower boundary condition set to 'natural'"
            d2[0] = 0.
            temp[0] = 0.
        else:
            d2[0] = -0.5
            temp[0] = 3./(x[1]-x[0]) * ((y[1]-y[0])/(x[1]-x[0]) - firstderiv)

        temp = self._tridiag(temp,d2)

        if lastderiv is None:
            if verbose:  print "Upper boundary condition set to 'natural'"
            qn = 0.
            un = 0.
        else:
            qn = 0.5
            un = 3./(x[-1]-x[-2]) * (lastderiv - (y[-1]-y[-2])/(x[-1]-x[-2]))

        d2[-1] = (un - qn*temp[-1]) / (qn*d2[-2] + 1.)
        for i in reversed(range(npts-1)):
            d2[i] = d2[i] * d2[i+1] + temp[i]

        self.d2 = d2

def spline_continuum(wa, fl, er, edges, minfrac=0.01, nsig=3.0,
                      resid_std=1.3, debug=False, omitregions=None):
    """ Given a section of spectrum, fit a continuum to it very
    loosely based on the method in Aguirre et al. 2002.

    Shamelessly stolen from Neil Crighton's pyserpens on 31/7/2013

    Parameters
    ----------
    wa               : Wavelengths.
    fl               : Fluxes.
    er               : One sigma errors.
    edges            : Wavelengths giving the chunk edges.
    minfrac = 0.01   : At least this fraction of pixels in a single chunk
                       contributes to the fit.
    nsig = 3.0       : No. of sigma for rejection for clipping.
    resid_std = 1.3  : Maximum residual st. dev. in a given chunk.
    debug = False    : If True, make helpful plots.
    omitregions=None : Wavelength regions to omit (due to molecular bandheads
                       or poor telluric cancellation

    Returns
    -------
    Continuum array, spline points, first derivative at first and last
    spline points

    Examples
    --------
    """

    # Overview:

    # (1) Calculate the median flux value for each wavelength chunk.

    # (2) fit a 1st order spline (i.e. series of straight line
    # segments) through the set of points given by the central
    # wavelength for each chunk and the median flux value for each
    # chunk.

    # (3) Remove any flux values that fall more than nsig*er below
    # the spline.

    # Repeat 1-3 until the continuum converges on a solution (if it
    # doesn't throw hands up in despair! Essential to choose a
    # suitable first guess with small enough chunks).

    if len(edges) < 2:
        raise ValueError('must be at least two bin edges!')

    wa,fl,er = (numpy.asarray(a) for a in (wa,fl,er))

    if debug:
        ax = pyplot.gca()
        ax.cla()
        ax.plot(wa,fl)
        ax.plot(wa,er)
        ax.axhline(0, color='0.7')
        good = ~numpy.isnan(fl) & ~numpy.isnan(er)
        ymax = 2*sorted(fl[good])[int(len(fl[good])*0.95)]
        ax.set_ylim(-0.1*ymax, ymax)
        ax.set_xlim(min(edges), max(edges))
        ax.set_autoscale_on(0)
        pyplot.draw()

    npts = len(wa)
    mask = numpy.ones(npts, bool)
    oldco = numpy.zeros(npts, float)
    co = numpy.zeros(npts, float)

    if omitregions:
        for omit in omitregions:
            edges.append(omit[0])
            edges.append(omit[1])
        edges.sort()

    # find indices of chunk edges and central wavelengths of chunks
    indices = wa.searchsorted(edges)
    indices = [(i0,i1) for i0,i1 in zip(indices[:-1],indices[1:])]
    wavc = [0.5*(w1 + w2) for w1,w2 in zip(edges[:-1],edges[1:])]
    omitted_regions = []
    good_regions = []
    good_wl = []
    if omitregions:
        for i,j  in zip(wavc, indices):
            for omitted in omitregions:
                if (i > omitted[0]) & (i < omitted[1]):
                    omitted_regions.append(j)
                else:
                    good_regions.append(j)
                    good_wl.append(i)
        indices = good_regions
        wavc = good_wl
    if debug:  print ' indices',indices

    # information per chunks
    npts = len(indices)
    mfl = numpy.zeros(npts, float)     # median fluxes at chunk centres
    goodfit = numpy.zeros(npts, bool)  # is fit acceptable?
    res_std = numpy.zeros(npts, float) # residuals standard dev
    res_med = numpy.zeros(npts, float) # residuals median
    if debug:
        print 'chunk centres',wavc
        cont, = ax.plot(wa,co,'k')
        midpoints, = ax.plot(wavc,mfl,'rx',mew=1.5,ms=8)

    # loop that iterative fits continuum
    while True:
        for i,(j1,j2) in enumerate(indices):
            if goodfit[i]:  continue
            # calculate median flux
            #print i,j1,j2
            w,f,e,m = (item[j1:j2] for item in (wa,fl,er,mask))
            ercond = e > 0
            cond = m & ercond
            chfl = f[cond]
            chflgood = f[ercond]
            #print len(chfl), len(chflgood)
            #print asdf
            #raw_input()
            if len(chflgood) == 0: continue
            if float(len(chfl)) / len(chflgood) < minfrac:
                f_cutoff = percentile(chflgood[ercond], minfrac)
                cond = ercond & (f >= f_cutoff)
            if len(f[cond]) == 0:  continue
            mfl[i] = numpy.median(f[cond])
        # calculate the spline. add extra points on either end to give
        # a nice slope at the end points.
        extwavc = ([wavc[0]-(wavc[1]-wavc[0])] + wavc +
                   [wavc[-1]+(wavc[-1]-wavc[-2])])
        extmfl = ([mfl[0]-(mfl[1]-mfl[0])] + list(mfl) +
                  [mfl[-1]+(mfl[-1]-mfl[-2])])
        co = numpy.interp(wa,extwavc,extmfl)
        if debug:
            cont.set_ydata(co)
            midpoints.set_xdata(wavc)
            midpoints.set_ydata(mfl)
            pyplot.draw()

        # calculate residuals for each chunk
        for i,(j1,j2) in enumerate(indices):
            if goodfit[i]:  continue
            ercond = er[j1:j2] > 0
            cond = ercond & mask[j1:j2]
            chfl = fl[j1:j2][cond]
            chflgood = fl[j1:j2][ercond]
            if len(chflgood) == 0:  continue
            if float(len(chfl)) / len(chflgood) < minfrac:
                f_cutoff = percentile(chflgood[ercond], minfrac)
                cond = ercond & (fl[j1:j2] > f_cutoff)
            #print len(co), len(fl), i1, j1, j2
            residuals = (fl[j1:j2][cond] - co[j1:j2][cond]
                         ) / er[j1:j2][cond]
            res_std[i] = residuals.std()
            if len(residuals) == 0:
                continue
            res_med[i] = numpy.median(residuals)
            # If residuals have std < 1.0 and mean ~1.0, we might have
            # a reasonable fit.
            if res_std[i] <= resid_std:
                goodfit[i] = True

        if debug:
            print 'median and st. dev. of residuals by region - aiming for 0,1'
            for i,(f0,f1) in  enumerate(zip(res_med, res_std)):
                print '%s %.2f %.2f' % (i,f0,f1)
            raw_input('Enter...')

        # (3) Remove flux values that fall more than N*sigma below the
        # spline fit.
        cond = (co - fl) > nsig * er
        if debug:
            print numpy.nanmax(numpy.abs(co - oldco)/co)
        # Finish when the biggest change between the new and old
        # medians is smaller than the number below.
        if numpy.nanmax(numpy.abs(co - oldco)/co) < 4e-3:
            break
        oldco = co.copy()
        mask[cond] = False

    # finally fit a cubic spline through the median values to
    # get a smooth continuum.
    d1 = (mfl[1] - mfl[0]) / (wavc[1]-wavc[0])
    d2 = (mfl[-1] - mfl[-2]) / (wavc[-1]-wavc[-2])
    final = InterpCubicSpline(wavc, mfl, firstderiv=d1, lastderiv=d2)

    return final(wa), zip(wavc,mfl), (d1,d2)


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

    mn = numpy.median(flat[nostrong])
    sig = numpy.std(flat[nostrong])

    first_pass = scipy.where( (flat > mn-1.5*sig) & (flat < mn+2*sig) )[0]

    mn = numpy.mean(flat[first_pass])
    sig = numpy.std(flat[first_pass])

    second_pass = scipy.where( (flat > mn-sig) & (flat < mn+2*sig) )[0]
    #spectral_slope = spectralSlope(wl[second_pass], flat[second_pass],
    #     dFlux[second_pass], wlStart, wlStop, 0.0, **kwargs)

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

    spline = scipy.interpolate.UnivariateSpline(wl[second_pass],
            flat[second_pass]+sig, s=1)
    #sp = Gnuplot.Data(wl, spline(wl))
    #plt.plot(first, sp)
    #raw_input()
    #continuum = (spectral_slope[0]+sig)*(wl/wlStart)**spectral_slope[1]
    flat = flat/spline(wl)
    if 'errors' in kwargs:
        return wl, flat, second_pass
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
