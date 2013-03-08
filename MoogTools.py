import scipy
import scipy.interpolate
import numpy


class periodicTable( object ):
    def __init__(self):
        self.Zsymbol_table = {}
        df = open('/home/deen/Code/Python/AstroCode/MOOGConstants.dat', 'r')
        for line in df.readlines():
            l = line.split('-')
            self.Zsymbol_table[int(l[0])] = l[1].strip()
            self.Zsymbol_table[l[1].strip()] = int(l[0])
        df.close()

    def translate(self, ID):
        retval = self.Zsymbol_table[ID]
        return retval

class VALD_Line( object ):
    def __init__(self, line1, line2, pt):
        l1 = line1.split(',')
        l2 = line2.split()
        self.element = pt.translate(l1[0].strip('\'').split()[0])
        self.ionization = int(l1[0].strip('\'').split()[1])-1
        self.species = self.element + self.ionization/10.0
        self.wl = float(l1[1])
        self.loggf = float(l1[2])
        self.expot_lo = float(l1[3])
        self.J_lo = float(l1[4])
        self.expot_hi = float(l1[5])
        self.J_hi = float(l1[6])
        self.g_lo = float(l1[7])
        self.g_hi = float(l1[8])
        self.g_eff = float(l1[9])
        self.radiative = float(l1[10])
        self.stark = float(l1[11])
        self.VdW = float(l1[12])
        self.DissE = -99.0
        self.transition = line2.strip().strip('\'')

        if (self.g_lo == 99.0):
            if not (self.species in [70.1, 25.2]):
                angmom = {"S":0, "P":1, "D":2, "F":3, "G":4, "H":5, "I":6, "K":7, "L":8, "M":9}
                n = 0
                try:
                    for char in self.transition:
                        if char.isdigit():
                            S = (float(char)-1.0)/2.0
                        if ((char.isupper()) & (n < 2)):
                            n+=1
                            L = angmom[char]
                            if n == 1:
                                if (self.J_lo > 0.0):
                                    self.g_lo = 1.5+(S*(S+1.0)-L*(L+1))/(2*self.J_lo*(self.J_lo+1))
                                else:
                                    self.g_lo = 0.0
                            else:
                                if (self.J_hi > 0.0):
                                    self.g_hi = 1.5+(S*(S+1.0)-L*(L+1))/(2*self.J_hi*(self.J_hi+1))
                                else:
                                    self.g_hi = 0.0
                except:
                    self.g_lo = 0.0
                    self.g_hi = 0.0
                    print("Ooops!")
            else:
                self.g_lo = 0.0
                self.g_hi = 0.0
   
        self.lower = Observed_Level(self.J_lo, self.g_lo, self.expot_lo)
        self.upper = Observed_Level(self.J_hi, self.g_hi, self.expot_lo+12400.0/self.wl)
        self.zeeman = {}
        self.zeeman["NOFIELD"] = [[self.wl], [self.loggf]]

    def zeeman_splitting(self, B, **kwargs):
        self.compute_zeeman_transitions(B, **kwargs)
        wl = []
        lgf = []
        for transition in self.pi_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight*
                    10.0**(self.loggf)))
        self.zeeman["PI"] = [numpy.array(wl), numpy.array(lgf)]

        wl = []
        lgf = []
        for transition in self.lcp_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight*
                    10.0**(self.loggf)))
        self.zeeman["LCP"] = [numpy.array(wl), numpy.array(lgf)]

        wl = []
        lgf = []
        for transition in self.rcp_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight*
                    10.0**(self.loggf)))
        self.zeeman["RCP"] = [numpy.array(wl), numpy.array(lgf)]

    def compute_zeeman_transitions(self, B, **kwargs):

        bohr_magneton = 5.78838176e-5        #eV*T^-1
        hc = 12400                           #eV*Angstroms
        lower_energies = {}
        upper_energies = {}
        for mj in self.lower.mj:
            lower_energies[mj]=self.lower.E+mj*self.lower.g*bohr_magneton*B

        for mj in self.upper.mj:
            upper_energies[mj] = self.upper.E+mj*self.upper.g*bohr_magneton*B

        pi_transitions = []
        lcp_transitions = []
        rcp_transitions = []

        pi_weight = 0.0
        lcp_weight = 0.0
        rcp_weight = 0.0

        delta_J = self.upper.J - self.lower.J
        J1 = self.lower.J

        self.geff = (0.5*(self.lower.g+self.upper.g)
                +0.25*(self.lower.g-self.upper.g)*(self.lower.J*(self.lower.J+1)-
                self.upper.J*(self.upper.J+1.0)))

        for mj in lower_energies.keys():
            if (delta_J == 0.0):
                if upper_energies.has_key(mj+1.0):    #delta Mj = +1 sigma component
                    weight = (J1-mj)*(J1+mj+1.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight+=weight
                if upper_energies.has_key(mj):    #delta Mj = 0 Pi component
                    weight = mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj, mj))
                    pi_weight+=weight
                if upper_energies.has_key(mj-1.0):    #delta Mj = -1 sigma component
                    weight = (J1+mj)*(J1-mj+1.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight+=weight
            if (delta_J == 1.0):
                if upper_energies.has_key(mj+1.0):    #delta Mj = +1 sigma component
                    weight = (J1+mj+1.0)*(J1+mj+2.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight+=weight
                if upper_energies.has_key(mj):    #delta Mj = 0 Pi component
                    weight = (J1+1.0)**2.0 - mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj, mj))
                    pi_weight+=weight
                if upper_energies.has_key(mj-1.0):    #delta Mj = -1 sigma component
                    weight = (J1-mj+1.0)*(J1-mj+2.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight+=weight
            if (delta_J == -1.0):
                if upper_energies.has_key(mj+1.0):    #delta Mj = +1 sigma component
                    weight = (J1-mj)*(J1-mj-1.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight+=weight
                if upper_energies.has_key(mj):    #delta Mj = 0 Pi component
                    weight = J1**2.0 - mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj, mj))
                    pi_weight+=weight
                if upper_energies.has_key(mj-1.0):    #delta Mj = -1 sigma component
                    weight = (J1+mj)*(J1+mj-1.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight+=weight

        for transition in rcp_transitions:
            transition.weight /= rcp_weight
        for transition in lcp_transitions:
            transition.weight /= lcp_weight
        for transition in pi_transitions:
            transition.weight /= pi_weight

        self.pi_transitions = pi_transitions
        self.lcp_transitions = lcp_transitions
        self.rcp_transitions = rcp_transitions

    def dump(self, **kwargs):
        if "out" in kwargs:
            out = kwargs["out"]
            if kwargs["mode"].upper() == 'MOOG':
                if( (self.expot_lo < 20.0) & (self.species % 1 <= 0.2)):
                    if (self.DissE == -99.0):
                        for i in range(len(self.zeeman["PI"][0])):
                            out.write('%10.3f%10s%10.3f%10.5f' %
                               (self.zeeman["PI"][0][i],
                               self.species,self.expot_lo,self.zeeman["PI"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f'% (' ',0.0))
                            else:
                                out.write('%10.3f%20s%10.3f' %
                                        (self.VdW, ' ', 0.0))
                            if self.radiative == 0:
                                out.write('%10.3s'% (' '))
                            else:
                                out.write('%10.3f' %
                                        (self.radiative))
                            if self.stark == 0:
                                out.write('%10s\n'% (' '))
                            else:
                                out.write('%10.3f\n' %
                                        (self.stark))
                        for i in range(len(self.zeeman["LCP"][0])):
                            out.write('%10.3f%10s%10.3f%10.5f' %
                               (self.zeeman["LCP"][0][i],
                               self.species,self.expot_lo,self.zeeman["LCP"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f'% (' ',-1.0))
                            else:
                                out.write('%10.3f%20s%10.3f' %
                                        (self.VdW, ' ', -1.0))
                            if self.radiative == 0:
                                out.write('%10.3s'% (' '))
                            else:
                                out.write('%10.3f' %
                                        (self.radiative))
                            if self.stark == 0:
                                out.write('%10s\n'% (' '))
                            else:
                                out.write('%10.3f\n' %
                                        (self.stark))
                        for i in range(len(self.zeeman["RCP"][0])):
                            out.write('%10.3f%10s%10.3f%10.5f' %
                               (self.zeeman["RCP"][0][i],
                               self.species,self.expot_lo,self.zeeman["RCP"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f'% (' ',1.0))
                            else:
                                out.write('%10.3f%20s%10.3f' %
                                        (self.VdW, ' ', 1.0))
                            if self.radiative == 0:
                                out.write('%10.3s'% (' '))
                            else:
                                out.write('%10.3f' %
                                        (self.radiative))
                            if self.stark == 0:
                                out.write('%10s\n'% (' '))
                            else:
                                out.write('%10.3f\n' %
                                        (self.stark))
                    else:
                        #RCP
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.expot_lo,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f' %
                                    (' ',self.DissE, 1.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f' %
                                    (self.VdW, self.DissE, 1.0))
                        if self.radiative == 0:
                            out.write('%10.3s'% (' '))
                        else:
                            out.write('%10.3f' %
                                    (self.radiative))
                        if self.stark == 0:
                            out.write('%10s\n'% (' '))
                        else:
                            out.write('%10.3f\n' %
                                    (self.stark))
                        #PI
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.expot_lo,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f' %
                                    (' ',self.DissE, 0.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f' %
                                    (self.VdW, self.DissE, 0.0))
                        if self.radiative == 0:
                            out.write('%10.3s'% (' '))
                        else:
                            out.write('%10.3f' %
                                    (self.radiative))
                        if self.stark == 0:
                            out.write('%10s\n'% (' '))
                        else:
                            out.write('%10.3f\n' %
                                    (self.stark))
                        #LCP
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.expot_lo,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f' %
                                    (' ',self.DissE, -1.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f' %
                                    (self.VdW, self.DissE, -1.0))
                        if self.radiative == 0:
                            out.write('%10.3s'% (' '))
                        else:
                            out.write('%10.3f' %
                                    (self.radiative))
                        if self.stark == 0:
                            out.write('%10s\n'% (' '))
                        else:
                            out.write('%10.3f\n' %
                                    (self.stark))

class zeemanTransition( object):
    def __init__(self, wavelength, weight, m_up, m_low):
        self.wavelength = wavelength
        self.weight = weight
        self.m_up = m_up
        self.m_low = m_low

    def __eq__(self, other):
        return ( (self.wavelength == other.wavelength) &
                (self.m_up == other.m_up) & (self.m_low == other.m_low) )

class Observed_Level( object ):
    def __init__(self, J, g, E):
        self.E = E
        self.J = J
        if g != 99:
            self.g = g
        else:
            self.g = 1.0

        self.mj = numpy.arange(self.J, (-1.0*self.J)-0.5, step = -1)

def parse_VALD(VALD_list, strong_file, molecules, wl_start, wl_stop, Bfield):
    pt = periodicTable()

    strong = []
    for line in open(strong_file, 'r'):
        l = line.split()
        strong.append([float(l[0]), float(l[1])])
    
    vald_in = open(VALD_list, 'r')
    l1 = ''
    l2 = ''
    stronglines = []
    weaklines = []
    for line in vald_in:
        if line[0] != '#':
            if line[0] == '\'':
                l1 = line
            else:
                l2 = line
                current_line = VALD_Line(l1, l2, pt)
                wl = current_line.wl
                if ( (wl > wl_start) & (wl < wl_stop) ):
                    current_line.zeeman_splitting(Bfield)
                    species = current_line.species
                    if ( [wl, species] in strong):
                        stronglines.append(current_line)
                    else:
                        weaklines.append(current_line)

    return stronglines, weaklines


def rtint(mu, inten, deltav, vsini_in, vrt_in, **kwargs):
    """
    This is a python translation of Jeff Valenti's disk integration routine
    
    PURPOSE:
        Produces a flux profile by integrating intensity profiles (sampled
           at various mu angles) over the visible stellar surface.

    Calling Sequence:
        flux = rtint(mu, inten, deltav, vsini, vrt)

    INPUTS:
        MU: list of length nmu cosine of the angle between the outward normal
            and the line of sight for each intensity spectrum INTEN
        INTEN:  list (of length nmu) numpy arrays (each of length npts)
            intensity spectra at specified values of MU
        DELTAV: (scalar) velocity spacing between adjacent spectrum points in
            INTEN (same units as VSINI and VRT)

        VSIN (scalar) maximum radial velocity, due to solid-body rotation
        VRT (scalar) radial-tangential macroturbulence parameter, i.e.. sqrt(2)
            times the standard deviation of a Gaussian distribution of 
            turbulent velocities.  The same distribution function describes
            the raidal motions of one component and the tangential motions of
            a second component.  Each component covers half the stellar surface.
            See "Observation and Analysis of Stellar Photospheres" by Gray.

    INPUT KEYWORDS:
        OSAMP: (scalar) internal oversamping factor for the convolutions.  By
            default, convolutions are done using the input points (OSAMP=1), 
            but when OSAMP is set to higher integer values, the input spectra
            are first oversamping via cubic spline interpolation.

    OUTPUTS:
        function value: numpy array of length npts producing the disk-integrated
            flux profile.

    RESTRICTIONS:
        Intensity profiles are weighted by the fraction of the projected stellar
            surface they represent, apportioning the area between adjacent MU
            points equally.  Additional weights (such as those used in a Gauss-
            Legendre quadrature) cannot meaningfully be used in this scheme.
            About twice as many points are required with this scheme to achieve
            the same precision of Gauss-Legendre quadrature.
        DELTAV, VSINI, and VRT must all be in the same units (e.q. km/s).
        If specified, OSAMP should be a positive integer

    AUTHOR'S REQUEST:
        If you use this algorithm in work that you publish, please cite...

    MODIFICATION HISTORY:
            Feb 88  GM Created ANA version
         13 Oct 92 JAV Adapted from G. Marcy's ANA routine of same name
         03 Nov 93 JAV Switched to annular convolution technique
         12 Nov 93 JAV Fixed bug. Intensity components not added when vsini=0
         14 Jun 94 JAV Reformatted for "public" release.  Heavily commented.
                 Pass deltav instead of 2.998d5/deltav.  Added osamp
                    keyword.  Added rebinning logic and end of routine.
                 Changed default osamp from 3 to 1.
         20 Feb 95 JAV Added mu as an argument to handle arbitrary mu sampling
                    and remove ambiguity in intensity profile ordering.
                 Interpret VTURB as sqrt(2)*sigma instead of just sigma
                 Replaced call_external with call to spl_{init|interp}.
         03 Apr 95 JAV Multiply flux by !pi to give observed flux.
         24 Oct 95 JAV Force "nmk" padding to be at least 3 pixels
         18 Dec 95 JAV Renamed from dkint() to rtint().  No longer make local
                    copy of intensities.  Use radial-tangential instead of 
                    isotropic Gaussian macroturbulence.
         26 Jan 99 JAV For NMU=1 and VSINI=0, assume resolved solar surface;
                    apply R-T macro, but supress vsini broadening.
         01 Apr 99 GMH Use annuli weights, rather than assuming equal area.
         27 Feb 13 CPD Translated to Python

    """
    
    #make local copies of various input variables, which will be altered below
    vsini = float(vsini_in)
    vrt = float(vrt_in)

    if "OSAMP" in kwargs:
        os = max(round(kwargs["OSAMP"]), 1)
    else:
        os = 1

    #Convert input MU to projected radii, R of annuli for a star of unit radius
    #(which is just sine rather than cosine of the angle between the outward
    #normal and the LOS)
    rmu = numpy.sqrt(1.0-mu**2)

    #Sort the projected radii and corresponding intensity spectra into ascending
    #order (i.e. from disk center to the limb), which is equivalent to sorting
    #MU in decending order
    order = numpy.argsort(rmu)
    rmu = rmu[order]
    nmu = len(mu)
    if (nmu == 1):
        vsini = 0.0

    #Calculate the projected radii for boundaries of disk integration annuli.
    #The n+1 boundaries are selected such that r(i+1) exactly bisects the area
    #between rmu(i) and rmu(i+1).  The innermost boundary, r(0) is set to 0
    #(Disk center) and the outermost boundary r(nmu) is set to to 1 (limb).
    if ((nmu > 1) | (vsini != 0)):
        r = numpy.sqrt(0.5*(rmu[0:-1]**2.0+rmu[1:])).tolist
        r.insert(0, 0.0)
        r.append(1.0)
        r = numpy.array(r)

    #Calculate integration weights for each disk integration annulus.  The
    #weight is just given by the relative area of each annulus, normalized such
    #that the sum of all weights is unity.  Weights for limb darkening are
    #included explicitly in the intensity profiles, so they aren't needed here.
        wt = r[1:]**2.0 - r[0:-1]**2.0
    else:
        wt = numpy.array([1.0])
    
    #Generate index vectors for input and oversampled points.  Note that the
    #oversampled indicies are carefully chosen such that every "os" finely
    #sampled points fit exactly into one input bin.  This makes it simple to
    #"integrate" the finely sampled points at the end of the routine.

    npts = len(inten[0])
    xpix = numpy.arange(npts)
    nfine = os*npts
    xfine = 0.5/os * 2.0*numpy.arange(nfine)-os+1

    #Loop through annuli, constructing and convolving with rotation kernels.
    dummy = 0
    yfine = numpy.zeros(nfine)
    flux = numpy.zeros(nfine)
    for m, y, w, i in zip(mu, inten, wt, range(nmu)):
        #use cubic spline routine to make an oversampled version of the
        #intensity profile for the current annulus.
        if os== 1:
            yfine = y.copy()
        else:
            yspl = scipy.interpolate.splrep(xpix, y)
            yfine = scipy.interpolate.splev(yspl, xfine)

    # Construct the convolution kernel which describes the distribution of 
    # rotational velocities present in the current annulus.  The distribution
    # has been derived analyitically for annuli of arbitrary thickness in a 
    # rigidly rotating star.  The kernel is constructed in two places: one 
    # piece for radial velocities less than the maximum velocity along the
    # inner edge of the annulus, and one piece for velocities greater than this
    # limit.
        if vsini < 0:
            r1 = r(i)
            r2 = r(i+1)
            dv = deltav/os
            maxv = vsini * r2
            nrk = 2*long(maxv/dv) + 3
            v = dv * (numpy.array(nrk) - ((nrk-1)/2.))
            rkern = numpy.zeros(nrk)
            j1 = scipy.where(abs(v) < vsini*r1)
            if len(j1[0]) > 0:
                rkern[j1] = (numpy.sqrt((vsini*r2)**2 - v[j1]**2)-
                        numpy.sqrt((vsini*r1)**2 - v[j1]**2))
            j2 = scipy.where((abs(v) >= vsini*r1) & (abs(v) <= vsini*r2))
            if len(j2[0]) > 0:
                rkern[j2] = numpy.sqrt((vsini*r2)**2 - v[j2]**2)
            rkern = rkern / rkern.sum()   # normalize kernel


    # Convolve the intensity profile with the rotational velocity kernel for
    # this annulus.  Pad the end of each profile with as many points as are in
    # the convolution kernel.  This reduces Fourier ringing.  The convolution 
    # may also be done with a routine called "externally" which efficiently
    # shifts and adds.
            if nrk > 3:
                yfine = scipy.convolve(yfine, rkern)

    # Calculate projected simga for radial and tangential velocity distributions.
        sigma = os*vrt/numpy.sqrt(2.0) /deltav
        sigr = sigma * m
        sigt = sigma * sqrt(1.0 - m**2.)

    # Figure out how many points to use in macroturbulence kernel
        nmk = max(min(round(sigma*10), (nfine-3)/2), 3)

    # Construct radial macroturbulence kernel w/ sigma of mu*VRT/sqrt(2)
        if sgr > 0:
            xarg = (numpy.range(2*nmk+1)-nmk) / sigr   # exponential arg
            mrkern = numpy.exp(max((-0.5*(xarg**2)),-20.0))
            mrkern = mrkern/mrkern.sum()
        else:
            mrkern = numpy.zeros(2*nmk+1)
            mrkern[nmk] = 1.0    #delta function

    # Construct tangential kernel w/ sigma of sqrt(1-mu**2)*VRT/sqrt(2.)
        if sigt > 0:
            xarg = (numpy.range(2*nmk+1)-nmk) /sigt
            mtkern = exp(max((-0.5*(xarg**2)), -20.0))
            mtkern = mtkern/mtkern.sum()
        else:
            mtkern = numpy.zeros(2*nmk+1)
            mtkern[nmk] = 1.0

    # Sum the radial and tangential components, weighted by surface area
        area_r = 0.5
        area_t = 0.5
        mkern = area_r*mkern + area_t*mtkern

    # Convolve the total flux profiles, again padding the spectrum on both ends 
    # to protect against Fourier rinnging.
        yfine = scipy.convolve(yfine, mkern)

    # Add contribution from current annulus to the running total
        flux += w*yfine

    return flux
