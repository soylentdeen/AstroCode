import numpy
import scipy
from scipy import interpolate
from scipy import integrate

def SpT2TeX(SpT, eunc, lunc):
    ''' Converts Spectral Types to LaTeX format '''
    if (SpT < 10.0):
        lett = 'O'
        num = str('%.1f' % SpT)
    elif (SpT < 20.0):
        lett = 'B'
        num = str('%.1f' % (SpT-10.0))
    elif (SpT < 30.0):
        lett = 'A'
        num = str('%.1f' % (SpT-20.0))
    elif (SpT < 40.0):
        lett = 'F'
        num = str('%.1f' % (SpT-30.0))
    elif (SpT < 50.0):
        lett = 'G'
        num = str('%.1f' % (SpT-40.0))
    elif (SpT < 58.0):
        lett = 'K'
        num = str('%.1f' % (SpT-50.0))
    elif (SpT < 70.0):
        lett = 'M'
        num = str('%.1f' % (SpT-58.0))

    spectral_type = lett+'$'+num+'_{-'+str('%.1f' % eunc)+'}^{+'+str('%.1f' % lunc)+'}$'

    if (SpT < 0):
        spectral_type = 'U'

    return spectral_type


def PMS_temp(SpT):
    ST = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0]
    Teff = [30000., 25400., 22000., 18700., 17000., 15400., 14000., 13000., 11900., 10500., 9520., 9230., 8970., 8720., 8460., 8200., 8350., 7850., 7580., 7390., 7200., 7050., 6890., 6740., 6590., 6440., 6360., 6280., 6200., 6115., 6030., 5945., 5860., 5830., 5800., 5770., 5700., 5630., 5520., 5410., 5250., 5080., 4900., 4730., 4590., 4350., 4205., 4060., 3850., 3705., 3560., 3415., 3270., 3125., 2990., 1000.]

    temp = scipy.interp(SpT, ST, Teff)
    return temp


def calc_photosphere(SpT):
    """ Returns the J-H, H-K colors of the photosphere """

    """ Have I included the conversion to 2MASS photometry?  I'm not sure if I have... """
    st = numpy.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67])
    vk = numpy.array([-0.93, -0.81, -0.74, -0.61, -0.55, -0.57, -0.43, -0.39, -0.35, -0.18, 0.00, 0.07, 0.14, 0.22, 0.30, 0.38, 0.44, 0.50, 0.57, 0.64, 0.70, 0.76, 0.82, 0.91, 1.01, 1.01, 1.21, 1.32, 1.35, 1.38, 1.41, 1.44, 1.46, 1.49, 1.53, 1.58, 1.64, 1.72, 1.76, 1.80, 1.96, 2.09, 2.22, 2.42, 2.63, 2.85, 3.00, 3.16, 3.65, 3.87, 4.11, 4.65, 5.26, 6.12, 7.30, 8.30])
    vj = numpy.array([-0.70, -0.61, -0.55, -0.45, -0.40, -0.35, -0.32, -0.29, -0.26, -0.14, 0.00, 0.06, 0.12, 0.18, 0.25, 0.30, 0.34, 0.39, 0.45, 0.50, 0.54, 0.58, 0.63, 0.69, 0.76, 0.83, 0.87, 0.98, 1.00, 1.03, 1.05, 1.08, 1.09, 1.11, 1.15, 1.16, 1.18, 1.27, 1.28, 1.30, 1.43, 1.53, 1.63, 1.79, 1.95, 2.13, 2.25, 2.37, 2.79, 3.00, 3.24, 3.78, 4.38, 5.18, 6.27, 7.27])
    vh = numpy.array([-0.81, -0.71, -0.65, -0.53, -0.47, -0.41, -0.37, -0.34, -0.31, -0.16, 0.00, 0.06, 0.13, 0.21, 0.28, 0.36, 0.41, 0.47, 0.54, 0.61, 0.67, 0.73, 0.79, 0.87, 0.97, 1.06, 1.17, 1.27, 1.30, 1.33, 1.36, 1.39, 1.41, 1.44, 1.47, 1.52, 1.58, 1.66, 1.69, 1.73, 1.88, 2.0, 2.13, 2.33, 2.53, 2.74, 2.88, 3.03, 3.48, 3.67, 3.91, 4.40, 4.98, 5.80, 6.93, 7.93])
    
    vj_0 = scipy.interp(SpT, st, vj)
    vh_0 = scipy.interp(SpT, st, vh)
    vk_0 = scipy.interp(SpT, st, vk)

    jh = vh_0 - vj_0
    hk = vk_0 - vh_0

    return jh, hk


def K_Bolcorr(SpT):
    """ Returns the K-band bolometric correction, given a spectral type """
    st = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67]
    bc = [-3.16, -2.70, -2.35, -1.94, -1.70, -1.46, -1.21, -1.02, -0.80, -0.51, -0.30, -0.23, -0.20, -0.17, -0.16, -0.15, -0.13, -0.12, -0.10, -0.10, -0.09, -0.10, -0.11, -0.12, -0.13, -0.14, -0.15, -0.16, -0.16, -0.17, -0.18, -0.19, -0.20, -0.20, -0.21, -0.21, -0.22, -0.23, -0.25, -0.28, -0.31, -0.37, -0.42, -0.50, -0.55, -0.72, -0.82, -0.92, -1.25, -1.43, -1.64, -2.03, -2.56, -3.29, -4.35, -4.35]
    vk = [-0.93, -0.81, -0.74, -0.61, -0.55, -0.57, -0.43, -0.39, -0.35, -0.18, 0.00, 0.07, 0.14, 0.22, 0.30, 0.38, 0.44, 0.50, 0.57, 0.64, 0.70, 0.76, 0.82, 0.91, 1.01, 1.01, 1.21, 1.32, 1.35, 1.38, 1.41, 1.44, 1.46, 1.49, 1.53, 1.58, 1.64, 1.72, 1.76, 1.80, 1.96, 2.09, 2.22, 2.42, 2.63, 2.85, 3.00, 3.16, 3.65, 3.87, 4.11, 4.65, 5.26, 6.12, 7.30, 7.30]
    vj = [-0.70, -0.61, -0.55, -0.45, -0.40, -0.35, -0.32, -0.29, -0.26, -0.14, 0.00, 0.06, -0.12, 0.18, 0.25, 0.30, 0.34, 0.39, 0.45, 0.50, 0.54, 0.58, 0.63, 0.69, 0.76, 0.83, 0.87, 0.98, 1.00, 1.03, 1.05, 1.08, 1.09, 1.11, 1.15, 1.16, 1.18, 1.27, 1.28, 1.30, 1.43, 1.53, 1.63, 1.79, 1.95, 2.13, 2.25, 2.37, 2.79, 3.00, 3.24, 3.78, 4.38, 5.18, 6.27, 6.27]

    bolcor = 0.0
    v_k = 0.0
    v_j = 0.0

    v_k = scipy.interp(SpT, st, vk)
    v_j = scipy.interp(SpT, st, vj)
    j_k = v_k - v_j
    bolcor = scipy.interp(SpT, st, bc)

    retval = 0.0

    if (SpT > 56.0):
         retval = j_k + 0.10*(v_k) + 1.17
    else:
         retval = v_k + bolcor
    
    return retval
    #return  (4.83+5*numpy.log10(12.0) - retval)/2.5


def photflux(x, y, filter):
    """ Provides synthetic photometry on the given spectrum using 2mass filter profiles """

    '''
    input:
       x         :  wavelength array (in nanometers)
       y         :  flux array
       filter    :  "2massj", "2massh", "2massk" are valid selections

    output:
       effective flux in the photometric pass-band.  Flux convolved with the filter profile/filter profile
    '''
    fdir = '/home/deen/Data/StarFormation/Photometry/'
    if filter in '2massj':
        fname = 'FILTER_PROFILES/J_2MASS.dat'
        fnuzero = 1594.0
        flzero = 3.129e-10  #erg/s/cm^2/Angstrom
        l_0 = 1228.5
        nu_0 = 3e18/l_0
        mstd = -0.001
    elif filter in '2massh':
        fname = 'FILTER_PROFILES/H_2MASS.dat'
        fnuzero = 1024.0
        flzero = 1.133e-10
        l_0 = 1638.5
        nu_0 = 3e18/l_0
        mstd = +0.019
    elif filter in '2massk':
        fname = 'FILTER_PROFILES/K_2MASS.dat'
        fnuzero = 666.7
        flzero = 4.283e-11
        l_0 = 2152.1
        nu_0 = 3e18/l_0
        mstd = -0.017

    # Opens and reads in the filter profile
    f = open(fdir+fname, 'r')
    dat = f.read()
    wds = numpy.array(dat.split(), float)
    npts = int(len(wds)/2.0)
    bm = numpy.arange(0,npts)
    minx = float(wds[0])*1e3
    maxx = float(wds[2*(npts-1)])*1e3
    fy = numpy.array(wds[2*bm+1], float)
    fy = fy/fy.max()
    fx = numpy.array(wds[2*bm], float)*1e3

    # trims the spectrum to only the areas covered by the filter
    bm = numpy.logical_and(x > minx,x < maxx).nonzero()
    fnew = scipy.interpolate.spline(fx, fy, x[bm])

    # Removes negative values in the filter profile
    neg = (fnew < 0.0).nonzero()
    fnew[neg] = 0.0

    # Computes the average flux over the wavelength region for the filter
    numerator = scipy.integrate.simps(y[bm]*fnew, x[bm])
    denom = scipy.integrate.simps(fnew, x[bm])
    retval = numerator/denom
    return retval

def planck_lambda(x, T):
    """ Blackbody function, returning f_lambda """
    h = 6.626e-27
    c = 2.998e10
    k = 1.381e-16
    xcm = x*1e-7
    nu = c/xcm
    retval = 2*h*c**2/xcm**5 *(1/(numpy.exp(h*c/(xcm*k*T)) - 1))
    return numpy.array(retval)

def planck_nu(x, T):
    """ Blackbody function, returning f_nu """
    h = 6.626e-27
    c = 2.998e10
    k = 1.381e-16
    xcm = x*1e-7
    nu = c/xcm
    retval = 2*h*nu**3/c**2 *(1/(numpy.exp(h*nu/(k*T)) - 1))
    return numpy.array(retval)

def excess_behavior(T, rk_limit):
    """ calculates behavior of J and H band excess vs K band excess for a Teff=T blackbody, shown in Cieza et al. 2005 """
    
    peakwave = (2.9e-3/T)*1e9    # Finds the peak wavelength (in nm)
    wave = numpy.linspace(100,5*peakwave,501)   # creates a suitable wavelength array (in nm)

    flambda = planck_nu(wave, T)
    tmj = photflux(wave, flambda, '2massj')
    tmh = photflux(wave, flambda, '2massh')
    tmk = photflux(wave, flambda, '2massk')
    
    rk = numpy.linspace(0, rk_limit, 2)
    Kexp = tmk/rk
    K = tmk/Kexp
    H = tmh/Kexp
    J = tmj/Kexp

    return J, H, K

def dered(J, H, K, dJ, dH, dK, beta):
    """ Dereddens the photometry to the T-Tauri locus using the beta supplied. """

    xj = 1.235
    xh = 1.662
    xk = 2.159

    Mctts = 0.63046
    Mred = (1-(xh/xj)**beta)/((xh/xj)**beta-(xk/xj)**beta)

    JH = J-H
    HK = H-K
    y1 = Mctts*HK + 0.4968
    x1 = (JH - y1)/(Mred - Mctts)
    denom = ((xh/xj)**beta-(xk/xj)**beta)

    Aj = x1/denom
    dAj = ( (dJ/denom)**2.0 + (dH/denom)**2.0 + (dK/denom)**2.0 )**(0.5)

    return Aj, dAj

