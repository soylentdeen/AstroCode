import scipy.signal
import scipy.interpolate
import scipy.optimize
import scipy.integrate
import numpy
import pyfits
import string

def resample(x, y, R):
    ''' This routine convolves a given spectrum to a resolution R'''

    subsample = 16.0

    xstart = x[0]
    xstop = x[-1]

    newx = [xstart]
    while newx[-1] < xstop:
        stepsize = newx[-1]/(R*subsample)
        newx.append(newx[-1]+stepsize)

    f = scipy.interpolate.interpolate.interp1d(x, y, bounds_error=False)
    newy = f(newx)
    const = numpy.ones(len(newx))

    xk = numpy.array(range(4.0*subsample))
    yk = numpy.exp(-(xk-(2.0*subsample))**2.0/(subsample**2.0/(4.0*numpy.log(2.0))))
    
    result = scipy.signal.convolve(newy, yk, mode ='valid')
    normal = scipy.signal.convolve(const, yk, mode = 'valid')

    bm = numpy.isfinite(result)
    return newx[int(len(xk)/2.0):-int(len(xk)/2.0)], result[bm]/normal[bm]


def write_2col_spectrum(filename, wl, fl):
    '''
    Prints a spectrum to a two-column data file
    '''

    data = open(filename, 'w')

    for line in zip(wl, fl):
        data.write(str(line[0])+' '+str(line[1])+'\n')

    data.close()

def write_MOOG_obs_spectrum(filename, wl, fl):
    '''
    Prints a spectrum to a two-column data file in a format useful for the MOOG spectral synthesis program.
    '''

    data = open(filename, 'w')

    for d in zip(wl, fl):
        data.write('%10f %9f\n' % (d[0], d[1]) )

    data.close()

def read_2col_spectrum(filename):
    '''
    Reads in a spectrum from a text file in a 2 column format.

    column 1: wavelength
    column 2: flux
    '''
    data = open(filename).read().split('\n')
    
    x = []
    y = []
    
    for line in data:
        l = line.split()
        if len(l) == 2:
            x.append(float(l[0]))
            y.append(float(l[1]))
            
    x = numpy.array(x)
    y = numpy.array(y)
    
    return x, y

def read_fits_spectrum(filename):
    hdulist = pyfits.open(filename, ignore_missing_end=True)
    hdr = hdulist[0].header
    dat = hdulist[0].data

    wl = dat[0]
    fl = dat[1]
    dfl = dat[2]
    
    bm = scipy.where( numpy.isfinite(fl) == True)

    return (wl[bm], fl[bm], dfl[bm])

def read_IRAF_fits_spectrum(filename):
    """ Reads in an echelle spectrum which has been reduced by IRAF """
    hdulist = pyfits.open(filename, ignore_missing_end = True)
    hdr = hdulist[0].header
    dat = hdulist[0].data

    "Finds the number of orders"
    n_orders = int(hdr["NAXIS2"])
    waveTable = hdr["WAT2*"]

    "Strings together the wavelength conversion strings"
    linear_header = ''
    for tableEntry in waveTable:
        linear_header += string.ljust(tableEntry.value, 68)

    """
    Extracts the coefficients necessary for the wavelength solution
    for each order
    """
    wlsol = []
    for sol in linear_header.split('spec')[2:]:
        sol = sol.split()
        wlsol.append([float(sol[5]), float(sol[6])])

    """
    Creates the wavelength solution for each pixel, and places it in a
    parallel array.
    """
    orders = []
    for i in range(len(dat[0])):
        wl = numpy.arange(len(dat[0][i]))*wlsol[i][1]+wlsol[i][0]
        orders.append([wl, dat[0][i]])

    orders = numpy.array(orders)
    return hdr, orders

def fit_gaussians(x, y, linecenters, R, **kwargs):

    params = []
    strength = -0.05
    for i in range(len(linecenters)):
        fwhm = 0.05*linecenters[i]/R
        if "strengthGuesses" in kwargs:
            params.append(kwargs["strengthGuesses"][i])
        else:
            params.append(strength)      #Strength
        params.append(linecenters[i])
        params.append(fwhm)          #FWHM

    def fitfunc(pars, xpts):
        retval = numpy.ones(len(xpts))
        for i in range(len(linecenters)):
            k = i*3
            for j in range(len(xpts)):
                retval[j] += pars[k]*numpy.exp(-(xpts[j]-pars[k+1])**2.0/(pars[k+2]))
        return retval

    def errfunc(pars, xpts, ypts):
        return numpy.abs(fitfunc(pars, xpts) - ypts)
        
    pfit = scipy.optimize.leastsq(errfunc, params, args=(x, y))

    coeffs = pfit[0]

    fit = numpy.ones(len(x))
    for i in range(len(linecenters)):
        k = i*3
        for j in range(len(x)):
            fit[j] += coeffs[k]*numpy.exp(-(x[j]-coeffs[k+1])**2.0/(coeffs[k+2]))

    return coeffs, fit

def calc_EW(x, y, xstart, xstop):
   bm = scipy.where( (x > xstart) & (x < xstop) )[0]
   cont = numpy.ones(len(bm))
   num = scipy.integrate.simps(y[bm], x[bm])
   denom = scipy.integrate.simps(cont, x[bm])
   return (denom-num)


def blackBody(**kwargs):
    """ Returns a blackbody function over the given wavelength  """

    """
    inputs:
        wl : wavelength array
               Assumed to be in units of cm, unless specified by wlUnits kwarg

        nu : frequency array
               
        T : Blackbody Temperature (K)

        wlUnits: Units of wavelengths (Optional)
               'Angstroms'
               'nanometers'
               'microns'
               'cm'
               'meters'
        outUnits: cgs Units to output.
               'Fnu'
               'Flambda'
               'Energy'

    outputs:
        y : Blackbody function.  Unit is assumed to be Flambda or Fnu (depnding on whether
                wl or nu was given) unless overridden by outUnits kwarg
    """
    h = 6.626e-27
    c = 2.998e10
    k = 1.38e-16
    T = kwargs["T"]

    if "wl" in kwargs:
        wl = kwargs["wl"]
        c1 = 2.0*h*c**2.0
        c2 = h*c/(k*T)
        Flambda = c1/(wl**5.0*(numpy.exp(c2/wl)-1.0))
        if "outUnits" in kwargs:
            if kwargs["outUnits"] == "Energy":
                return Flambda*wl
        else:
            return Flambda
    elif "nu" in kwargs:
        nu = kwargs["nu"]
        c1 = 2.0*h/(c**2.0)
        c2 = h/(k*T)
        Fnu = c1*nu**2.0/(numpy.exp(c2*nu) - 1.0)
        if "outUnits" in kwargs:
            if kwargs["outUnits"] == "Energy":
                return Fnu*nu
        else:
            return Fnu


class photometrySynthesizer( object ):
    def __init__(self, **kwargs):
        if "filterDir" in kwargs:
            self.fdir = filter_dir
        else:
            self.fdir = '/home/deen/Data/StarFormation/Photometry/FILTER_PROFILES/'

        filterNames = ['Uj', 'Bj', 'Vj', 'Rc', 'Ic', '2massj', '2massh', '2massk']
        fileNames = ['U_Landolt.dat', 'B_Bessell.dat', 'V_Bessell.dat', 'cousins_Rband.dat', 'cousins_Iband.dat', 'J_2MASS.dat', 'H_2MASS.dat', 'K_2MASS.dat']
        fnu_zero = [1829, 4144, 3544, 2950, 2280.0, 1594.0, 1024.0, 666.7 ]
        flam_zero = [4.0274905e-09, 6.3170333e-09, 3.6186341e-09, 2.1651655e-9, 1.1326593e-09, 3.129e-10, 1.133e-10, 4.283e-11] #erg/s/cm^2/Angstrom
        lambda_eff = [3600, 4362, 5446, 6413, 7978, 12285, 16385, 21521]
        mVega = [0.02, 0.02, 0.03, 0.039, 0.035, -0.001, +0.019, -0.017]

        self.photBands = []
        for band in zip(filterNames, fileNames, fnu_zero, flam_zero, lambda_eff, mVega):
            photBand = dict()
            photBand['Name'] = band[0]
            photBand['file'] = band[1]
            photBand['fnu_zero'] = band[2]
            photBand['flam_zero'] = band[3]
            photBand['lambda_eff'] = band[4]
            photBand['mVega'] = band[5]
            
            fx = []
            fy = []
            dat = open(self.fdir+band[1], 'r').read().split('\n')
            for line in dat:
                if len(line) > 0:
                    l = line.split()
                    fx.append(float(l[0])*1e4)
                    fy.append(float(l[1]))
            fy = numpy.array(fy)
            fx = numpy.array(fx)
            photBand['min_x'] = min(fx)
            photBand['max_x'] = max(fx)
            photBand['photSpline'] = scipy.interpolate.splrep(fx, fy)

            self.photBands.append(photBand)
    
    def photFlux(self, x, y, filtName):
        for band in self.photBands:
            if band['Name'] == filtName:
                bm = scipy.where( (x > band['min_x'] ) & (x < band['max_x']) )[0]
                fnew = scipy.interpolate.splev(x[bm], band['photSpline'])
                valid_bm = scipy.where( (fnew > 0.0) & (y[bm] > 0.0) )[0]
                numerator = scipy.integrate.simps(y[bm][valid_bm]*fnew[valid_bm], x[bm][valid_bm])
                denom = scipy.integrate.simps(fnew[valid_bm], x[bm][valid_bm])
                return numerator/denom
