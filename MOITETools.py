import numpy
import scipy

class QFactorCalculator( object ):
    def __init__(self, **kwargs):
        if "filterDir" in kwargs:
            self.fdir = filter_dir
        else:
            self.fdir = '/home/deen/Data/StarFormation/Photometry/FILTER_PROFILES/'
            
        filterNames = ['Uj', 'Bj', 'Vj', 'Rc', 'Ic', '2massj', '2massh','2massk']
        fileNames = ['U_Landolt.dat', 'B_Bessell.dat', 'V_Bessell.dat', 'cousins_Rband.dat', 'cousins_Iband.dat',
        'J_2MASS.dat','H_2MASS.dat', 'K_2MASS.dat']
        fnu_zero = [1829, 4144, 3544, 2950, 2280.0, 1594.0, 1024.0, 666.7 ]
        flam_zero = [4.0274905e-09, 6.3170333e-09, 3.6186341e-09, 2.1651655e-9, 1.1326593e-09, 3.129e-10, 1.133e-10,
        4.283e-11] #erg/s/cm^2/Angstrom
        lambda_eff = [3600, 4362, 5446, 6413, 7978, 12285, 16385, 21521]
        mVega = [0.02, 0.02, 0.03, 0.039, 0.035, -0.001, +0.019,-0.017]
        
        self.photBands = []
        for band in zip(filterNames, fileNames, fnu_zero, flam_zero, lambda_eff, mVega):
            photBand = dict()
            photBand['Name']= band[0]
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

    def calcQFactor(self, x, y, filtName):
        for band in self.photBands:
            if band['Name'] == filtName:
                bm = scipy.where( (x > band['min_x'] ) & (x < band['max_x']) )[0]
                fnew = scipy.interpolate.splev(x[bm], band['photSpline'])
                valid_bm = scipy.where( (fnew > 0.0) & (y[bm] > 0.0) )[0]
                mid = scipy.where( abs(x[bm][valid_bm] - band['lambda_eff']) < band['lambda_eff']/50.0)[0]

                numerator = scipy.integrate.simps(fnew[valid_bm], x[bm][valid_bm])
                denom = scipy.integrate.simps(y[bm][valid_bm]*fnew[valid_bm],
                x[bm][valid_bm])/numpy.median(y[bm][valid_bm][mid])
                return numerator/denom




class QFactors( object ):
    def __init__(self, qf):
        self.qf_all_teff = numpy.array(qf[0][0])
        self.qf_all_logg = numpy.array(qf[0][1])
        self.qf_all_mh = numpy.array(qf[0][2])
        self.metallicities = numpy.sort(numpy.unique(self.qf_all_mh))
        self.log_gs = numpy.sort(numpy.unique(self.qf_all_logg))
        self.splines = []
        for mh in self.metallicities:
            subset = []
            for logg in self.log_gs:
                bm = scipy.where( (self.qf_all_logg == logg) & (self.qf_all_mh == mh) )[0]
                order = numpy.argsort( self.qf_all_teff[bm] )
                qf_teff = self.qf_all_teff[bm][order]
                qf_U = scipy.interpolate.splrep(qf_teff, qf[0][3][bm][order])
                qf_B = scipy.interpolate.splrep(qf_teff, qf[0][4][bm][order])
                qf_V = scipy.interpolate.splrep(qf_teff, qf[0][5][bm][order])
                qf_R = scipy.interpolate.splrep(qf_teff, qf[0][6][bm][order])
                qf_I = scipy.interpolate.splrep(qf_teff, qf[0][7][bm][order])
                qf_J = scipy.interpolate.splrep(qf_teff, qf[0][8][bm][order])
                qf_H = scipy.interpolate.splrep(qf_teff, qf[0][9][bm][order])
                qf_K = scipy.interpolate.splrep(qf_teff, qf[0][10][bm][order])
                subset.append([qf_U, qf_B, qf_V, qf_R, qf_I, qf_J, qf_H, qf_K])
            self.splines.append(subset)

    def getQFactors(self, Teff, logg, feh):
        grav = scipy.where( (self.log_gs == logg) )[0]
        qfs = []
        for band in self.splines:
            b = []
            for spline in band[grav]:
                b.append(scipy.interpolate.splev(Teff, spline))
            qfs.append(b)

        retval = []
        #retval = [numpy.zeros(8, float), ['Uj','Bj','Vj','Rc','Ic','J','H','K']]
        for qf in zip(*qfs):
            retval.append(scipy.interpolate.spline(self.metallicities, qf, feh))

        retval = [numpy.array(retval), ['Uj','Bj','Vj','Rc','Ic','J','H','K']]
        return retval

class modFluxes( object ):
    def __init__(self, modfluxes):
        sigma = 5.67e-5
        G = 6.67259e-8
        M = 1.99e33
        self.factor = (G*M/10.0**(5.0))/(3.08568e19)**2.0
        self.teff_all_mod = numpy.array(modfluxes[0][0], float)
        self.logg_all_mod = numpy.array(modfluxes[0][1], float)
        self.mh_all_mod = numpy.array(modfluxes[0][2], float)
        self.mbol_all_mod = numpy.array(modfluxes[0][3], float)/self.factor
        self.bands = []
        for b in modfluxes[0][4:]:
            self.bands.append(numpy.array(b)/self.factor)
        self.bands = numpy.array(self.bands)

        self.metallicities = numpy.sort(numpy.unique(self.mh_all_mod))
        self.log_gs = numpy.sort(numpy.unique(self.logg_all_mod))

        self.splines = []
        for feh in self.metallicities:
            subset = []
            for logg in self.log_gs:
                bm = scipy.where( (self.logg_all_mod == logg) & (self.mh_all_mod == feh) )[0]
                order = numpy.argsort(self.teff_all_mod[bm])
                subsubset = []
                for i in range(len(self.bands)):
                    subsubset.append(scipy.interpolate.splrep(self.teff_all_mod[bm][order], self.bands[i][bm][order]))
                subset.append(subsubset)
            self.splines.append(subset)

    def getModFluxes(self, Teff, logg, feh):
        mfs = []
        grav = scipy.where( self.log_gs == logg) [0]
        for sp in self.splines:
            metal_level = []
            for spline in sp[grav]:
                metal_level.append(scipy.interpolate.splev(Teff, spline))
            mfs.append(metal_level)

        retval = []
        #retval = [numpy.zeros(8, float), ['Uj','Bj','Vj','Rc','Ic','J','H','K']]
        for mf in zip(*mfs):
            retval.append(scipy.interpolate.spline(self.metallicities, mf, feh))

        retval = numpy.array(retval)
        return retval

class MOITECalibration( object ):
    def __init__(self):
        colorName = ['V-Rc','V-Ic', 'V-J', 'V-H', 'V-Ks', 'Rc-Ic', 'Rc-J', 'Rc-H', 'Rc-Ks', 'Ic-J', 'Ic-H', 'Ic-Ks']
        colorRanges = [[0.800, 2.310], [1.400, 4.650], [2.260, 7.321], [2.946, 8.041], [3.219, 8.468], [0.660,2.270],
        [1.503, 5.374], [2.053, 6.001], [2.212, 6.428], [0.865, 2.954], [1.433, 3.644], [1.592, 4.085]]
        a0 = [-1.4095, 0.5050, 0.1926, -0.4711, -0.4809, 0.8326, 0.3594, -0.1645, -0.0570, -0.3813, 2.5844, -1.8798]
        a1 = [5.1212, 0.5562, 0.5738, 0.8450, 0.8009, 0.6122, 0.7223, 0.9269, 0.7737, 2.6488, 4.3925, 3.0706]
        a2 = [-2.7937, -0.0593, -0.0726, -0.1161, -0.1039, -0.0849, -0.1401, -0.1674, -0.1226, -1.1642, -1.5386, -0.9024]
        a3 = [0.5432, 0.0027, 0.0042, 0.0066, 0.0056, 0.0164, 0.0134, 0.0135, 0.0091, 0.1981, 0.1941, 0.0989]
        sig = [33, 26, 17, 23, 19, 27, 19, 31, 25, 37, 7, 61]
        
        self.calibrations = []
        for dat in zip(colorName, colorRanges, a0, a1, a2, a3, sig):
            calib = {'colorName':dat[0], 'colorRange':dat[1], 'a0':dat[2], 'a1':dat[3], 'a2':dat[4], 'a3':dat[5],
            'sig':dat[6]}
            self.calibrations.append(calib)

    def getTeffGuess(self, star):
        Teff_guesses = []
        sigmas = []
        for calib in self.calibrations:
            colors = calib['colorName'].split('-')
            c1 = colors[0]
            c2 = colors[1]
            if ( (c1 in star) & (c2 in star) ):
               if ( (star[c1] != -99.9) & (star[c2] != -99.9) ):
                   color = star[c1] - star[c2]
                   if ( (color > calib['colorRange'][0]) & (color < calib['colorRange'][1]) ):
                       Teff_guesses.append(5040.0/(calib['a0'] + calib['a1']*color +
                       calib['a2']*color**2.0+calib['a3']*color**3.0))
                       sigmas.append(calib['sig'])

        num = 0.0
        denom = 0.0
        for guess in zip(Teff_guesses, sigmas):
            num += guess[0]*guess[1]
            denom += guess[1]

        if (denom > 0):
            return num/denom
        else:
            return None
