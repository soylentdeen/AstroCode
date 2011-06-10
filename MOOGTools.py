import SpectralTools
import SEDTools
import scipy
import numpy
import scipy.integrate
import Gnuplot
import pickle


def write_parfile(filename, **kwargs):

    df = open(filename, 'w')
    labels = {'terminal':'x11', 'strong':1, 'atmosphere':1, 'molecules':2, 'lines':1, 'damping':1, 'freeform':0,
    'flux/int':0, 'plot':2, 'obspectrum':5}
    file_labels = {'summary_out':'summary.out', 'standard_out':'out1', 'smoothed_out':'smoothed.out', 'lines_in':'linelist.input',
    'stronglines_in':'stronglines.input', 'model_in':'model.md', 'observed_in':'observed.dat'}
    for l in labels:
        if l in kwargs:
            labels[l] = kwargs[l]

    for fl in file_labels:
        if fl in kwargs:
            file_labels[fl] = kwargs[fl]

    
    df.write(kwargs["mode"]+'\n')
    for fl in file_labels:
        df.write(fl+'      \''+file_labels[fl]+'\'\n')
    for l in labels:
        df.write(l+'        '+str(labels[l])+'\n')

    df.write('synlimits\n')
    df.write('               '+str(kwargs["wl_start"])+' '+str(kwargs["wl_stop"])+' 0.02 1.00\n')
    df.write('plotpars       1\n')
    df.write('               '+str(kwargs["wl_start"])+' '+str(kwargs["wl_stop"])+' 0.02 1.00\n')
    df.write('               0.00 0.000 0.000 1.00\n')
    df.write('               g 0.150 0.00 0.00 0.00 0.00\n')
    
    df.close()

class spectralSynthesizer_HR( object ):   # High resolution 

    def __init__(self):
        feat_num = [2, 57, 8]
        xstart = [1.190, 2.170, 2.260]
        xstop = [1.2100, 2.2112, 2.268]
        slope_start = [1.15, 2.100, 2.23]
        slope_stop = [1.25, 2.25, 2.29]
        continuum = [[1.2015, 1.2025], [2.1922, 2.1975], [2.258, 2.26]]
        strongLines = [[2.166, 2.2066], [2.166, 2.2066], [2.263, 2.267]]
        lineWidths = [[0.005, 0.005],[0.005, 0.005], [0.005, 0.005]]

        self.features = []

        for i in range(len(xstart)):
            feat = dict()
            print feat_num[i]
            feat["num"] = feat_num[i]
            feat["xstart"] = xstart[i]
            feat["xstop"] = xstop[i]
            feat["slope_start"] = slope_start[i]
            feat["slope_stop"] = slope_stop[i]
            feat["strongLines"] = strongLines[i]
            feat["lineWidths"] = lineWidths[i]
            feat["continuum"] = continuum[i]                    # Problably don't need this anymore
            feat["datadir"] = '/home/deen/Data/StarFormation/MOOG/features/feat_'+str(feat["num"])+'/MARCS_interpolated/'
            feat["models"], feat["wl"], feat["TandG"] = self.getMARCSModels(feat["num"])
            self.features.append(feat)

        temps = numpy.array([])
        gravs = numpy.array([])

        for feat in self.features:
            temps = numpy.append(temps, feat["TandG"][0])
            gravs = numpy.append(gravs, feat["TandG"][1])

        self.temps = numpy.unique(temps)
        self.gravs = numpy.unique(gravs)
        self.x_offsets = numpy.linspace(-10, 10, num = 21)  #wavelength offsets in angstroms
        self.y_offsets = numpy.linspace(1.00, 1.025, num=3)   #continuum offsets in % of continuum

        self.temps.sort()
        self.gravs.sort()

        self.coarse_t = numpy.arange(3900, 5400, 100)
        #self.coarse_t = self.temps[numpy.arange(0, len(self.temps), 2)]
        self.coarse_g = numpy.arange(300, 560, 20)
        #self.coarse_g = self.gravs[numpy.arange(0, len(self.gravs), 2)]
        self.coarse_xoff = self.x_offsets#[numpy.arange(0, len(self.x_offsets), 3)]
        self.coarse_yoff = numpy.array([1.000])
        #self.coarse_yoff = self.y_offsets#[numpy.arange(0, len(self.y_offsets), 4)]

    def getMARCSModels(self, fnum):
        filename = "feature_"+str(fnum)+"_models.pkl"

        data = pickle.load(open(filename) )
        models = data[0][0]
        wl = data[0][1]
        TandG = data[0][2]

        return models, wl, TandG

    def binMOOGSpectrum(self, spectrum, native_wl, new_wl):
        retval = numpy.zeros(len(new_wl))
        for i in range(len(new_wl)-1):
            bm = scipy.where( (native_wl > new_wl[i]) & (native_wl <= new_wl[i+1]) )[0]
            if (len(bm) > 0):
                num = scipy.integrate.simps(spectrum[bm], x=native_wl[bm])
                denom = max(native_wl[bm]) - min(native_wl[bm])
                retval[i] = num/denom
            else:
                retval[i] = retval[-1]

        bm = scipy.where (native_wl > new_wl[-1])
        num = scipy.integrate.simps(spectrum[bm], x=native_wl[bm])
        denom = max(native_wl[bm]) - min(native_wl[bm])
        retval[-1] = num/denom

        return retval

    def calcError(self, y1, y2, z):
        error = 0.0
        #obs = Gnuplot.Data(x, y1, with_='lines')
        #new = Gnuplot.Data(x, y2, with_='lines')
        #plt.plot(obs, new)
        for dat in zip(y1, y2, z):
            error += ((dat[0]-dat[1])/dat[2])**2
            
        return error/len(y1)


    def fitSpectrum(self, wl, flux, error, plt):   # wl in microns
        for feat in self.features:
            chisq = numpy.zeros([len(feat["TandG"][0]), len(self.x_offsets), len(self.y_offsets)])
            minchisq = 1e10
            x_window, flat, z = SEDTools.removeContinuum(wl, flux, error, feat["slope_start"], feat["slope_stop"],
            strongLines=feat["strongLines"], lineWidths=feat["lineWidths"], errors=True)
            x_sm = feat["wl"]/10000.0     # convert MOOG wavelengths to microns
            minx = min(x_sm)
            maxx = max(x_sm)
            for wl_offset in self.coarse_xoff:
                print wl_offset
                new_wl = x_window+wl_offset/10000.0       #microns
                overlap = scipy.where( (new_wl > minx) & (new_wl < maxx))[0]
                xoff_bm = scipy.where(self.x_offsets == wl_offset)[0]
                for fl_offset in self.coarse_yoff:
                    new_fl = flat*fl_offset
                    print fl_offset
                    yoff_bm = scipy.where(self.y_offsets == fl_offset)[0]
                    for T in self.coarse_t:
                        for G in self.coarse_g:
                            TandG_bm = scipy.where( (feat["TandG"][0] == T) & (feat["TandG"][1] == G) )[0]
                            if len(TandG_bm) > 0:
                                y_sm = feat["models"][TandG_bm]
                                synthetic_spectrum = self.binMOOGSpectrum(y_sm, x_sm, new_wl[overlap])
                                chisq[TandG_bm, xoff_bm, yoff_bm] = self.calcError(new_fl[overlap], synthetic_spectrum,
                                z[overlap])
                                if chisq[TandG_bm, xoff_bm, yoff_bm] < minchisq:
                                    minchisq = chisq[TandG_bm, xoff_bm, yoff_bm]
                                    print minchisq, T, G, fl_offset, wl_offset
                                    obs = Gnuplot.Data(new_wl[overlap], new_fl[overlap], with_='lines')
                                    syn = Gnuplot.Data(new_wl[overlap], synthetic_spectrum, with_='lines')
                                    plt.plot(obs, syn)

            #chisq = numpy.array(chisq)
            coarse_points = chisq.nonzero()
            coarse_chisq = chisq[coarse_points]
            order = numpy.argsort(coarse_chisq)
            chisq_ordered = coarse_chisq[order]
            T_ordered = numpy.array(feat["TandG"][0])[coarse_points[0][order]]
            G_ordered = numpy.array(feat["TandG"][1])[coarse_points[0][order]]
            wl_shift_ordered = self.x_offsets[coarse_points[1][order]]
            cont_shift_ordered = self.y_offsets[coarse_points[2][order]]
            print 'Minimum Chi-Squared : ', chisq_ordered[0:50]
            print 'Best-Fit Tempertures : ', T_ordered[0:50]
            print 'Best-Fit Surface Gravities : ', G_ordered[0:50]
            print 'Best-Fit Continuum Shift : ', cont_shift_ordered[0:50]
            print 'Best-Fit Wavelength Shift : ', wl_shift_ordered[0:50]
            mn_chisq = min(chisq_ordered)
            print 'Min Chi-squared : ', mn_chisq
            bm = scipy.where( (chisq_ordered - mn_chisq) < mn_chisq)[0]
            print 'Temperature range: ', numpy.mean(T_ordered[bm]), ' +/- ', numpy.std(T_ordered[bm])
            print 'Surface Gravity range: ', numpy.mean(G_ordered[bm]), ' +/- ', numpy.std(G_ordered[bm])
            print 'Wavelength Shift range: ', numpy.mean(wl_shift_ordered[bm]), ' +/- ', numpy.std(wl_shift_ordered[bm])
            print 'Continuum Scaling range: ', numpy.mean(cont_shift_ordered[bm]), ' +/- ', numpy.std(cont_shift_ordered[bm])
            print '1 sigma T range : ', 
            for i in range(10):
                new_wl = x_window+wl_shift_ordered[i]/10000.0
                overlap = scipy.where( (new_wl > minx) & (new_wl < maxx))[0]
                new_fl = flat*cont_shift_ordered[i]
                y_sm = feat["models"][coarse_points[0][order[i]]]
                synthetic_spectrum = self.binMOOGSpectrum(y_sm, x_sm, new_wl[overlap])
                obs = Gnuplot.Data(new_wl[overlap], new_fl[overlap], with_='lines')
                syn = Gnuplot.Data(new_wl[overlap], synthetic_spectrum, with_='lines')
                plt.plot(obs, syn)
                print wl_shift_ordered[i]
                raw_input()

class gridPoint( object ):    # Object which contains information about a Chi-Squared grid point
    def __init__(self, **kwargs):
        self.chisq = kwargs["chisq"]
        self.T = kwargs["T"]
        self.G = kwargs["G"]
        self.B = kwargs["B"]
        self.dx = kwargs["dx"]
        self.dy = kwargs["dy"]
        self.r = kwargs["r"]

class spectralSynthesizer( object ):   # low resolution

    def __init__(self):
        feat_num = [12, 57, 8]
        xstart = [1.190, 2.170, 2.260]
        xstop = [1.2100, 2.2112, 2.268]
        slope_start = [1.15, 2.100, 2.23]
        slope_stop = [1.25, 2.25, 2.29]
        strongLines = [[2.166, 2.2066], [2.166, 2.2066], [2.263, 2.267]]
        lineWidths = [[0.005, 0.005],[0.005, 0.005], [0.005, 0.005]]
        comparePoints = [[[1.1816,1.1838],[1.1871,1.1900],[1.1942,1.1995],[1.2073,1.2087]], [[2.1765, 2.18], [2.1863,
        2.1906], [2.199, 2.2015], [2.2037, 2.210]], [[2.2525,2.2551],[2.2592, 2.2669], [2.2796, 2.2818]]]

        self.modelBaseDir='/home/deen/Data/StarFormation/MOOG/zeeman/smoothed/'
        self.features = []

        #for i in range(len(xstart)):
        for i in [1]:
            feat = dict()
            print feat_num[i]
            feat["num"] = feat_num[i]
            feat["xstart"] = xstart[i]
            feat["xstop"] = xstop[i]
            feat["slope_start"] = slope_start[i]
            feat["slope_stop"] = slope_stop[i]
            feat["strongLines"] = strongLines[i]
            feat["lineWidths"] = lineWidths[i]
            feat["comparePoints"] = comparePoints[i]
            self.features.append(feat)

        self.temps = numpy.array(range(2500, 4000, 100)+range(4000,6250, 250))
        self.gravs = numpy.array(range(300, 600, 50))
        self.bfields = numpy.array(numpy.arange(0, 4.5, 0.5))

    def binMOOGSpectrum(self, spectrum, native_wl, new_wl):
        retval = numpy.zeros(len(new_wl))
        for i in range(len(new_wl)-1):
            bm = scipy.where( (native_wl > new_wl[i]) & (native_wl <= new_wl[i+1]) )[0]
            if (len(bm) > 0):
                num = scipy.integrate.simps(spectrum[bm], x=native_wl[bm])
                denom = max(native_wl[bm]) - min(native_wl[bm])
                retval[i] = num/denom
            else:
                retval[i] = retval[-1]

        bm = scipy.where (native_wl > new_wl[-1])[0]
        if len(bm) > 1:
            num = scipy.integrate.simps(spectrum[bm], x=native_wl[bm])
            denom = max(native_wl[bm]) - min(native_wl[bm])
            retval[-1] = num/denom
        else:
            retval[-1] = spectrum[bm]

        return retval

    def calcError(self, y1, y2, z, x, comparePoints):
        error = 0.0
        #obs = Gnuplot.Data(x, y1, with_='lines')
        #new = Gnuplot.Data(x, y2, with_='lines')
        #plt.plot(obs, new)
        bm = []
        for region in comparePoints:
            bm.extend(scipy.where( (x > region[0]) & (x < region[1]) )[0])
        for dat in zip(y1[bm], y2[bm], z[bm]):
            error += ((dat[0]-dat[1])/dat[2])**2
            
        return error/len(y1[bm])


    def findWavelengthShift(self, x_window, flat, x_sm, y_sm):
        orig_bm = scipy.where( (x_window > min(x_sm)) & (x_window < max(x_sm)) )[0]
        feature_x = x_window[orig_bm]
        model = scipy.interpolate.interpolate.interp1d(x_sm, y_sm, kind = 'linear', bounds_error = False)
        ycorr = scipy.correlate((1.0-flat[orig_bm]), (1.0-model(feature_x)), mode ='full')
        xcorr = scipy.linspace(0, len(ycorr)-1, num=len(ycorr))

        fitfunc = lambda p, x: p[0]*scipy.exp(-(x-p[1])**2/(2.0*p[2]**2)) + p[3]
        errfunc = lambda p, x, y: fitfunc(p, x) - y
        
        x_zoom = xcorr[len(ycorr)/2 - 3: len(ycorr)/2+5]
        y_zoom = ycorr[len(ycorr)/2 - 3: len(ycorr)/2+5]
        
        p_guess = [ycorr[len(ycorr)/2], len(ycorr)/2, 3.0, 0.0001]
        p1, success = scipy.optimize.leastsq(errfunc, p_guess, args = (x_zoom, y_zoom))
        
        fit = p1[0]*scipy.exp(-(x_zoom- p1[1])**2/(2.0*p1[2]**2)) + p1[3]
        
        xcorr = p1[1]
        nLags = xcorr-(len(orig_bm)-1.5)
        offset_computed = nLags*(feature_x[0]-feature_x[1])
        if abs(offset_computed) > 20:
            offset_computed = 0
            
        return offset_computed

    def calcVeiling(self, y_obs, y_calc, x, comparePoints):
        bm = []
        for region in comparePoints:
            bm.extend(scipy.where( (x > region[0]) & (x <region[1]) )[0])
            
        rk_guess = [0.01]
        ff = lambda p, y_c: (y_c+p[0])/(abs(p[0])+1.0)
        ef = lambda p, y_c, y_o: ff(p, y_c)-y_o
        veil, success = scipy.optimize.leastsq(ef, rk_guess, args = (y_calc[bm], y_obs[bm]))
        
        return veil[0]

    def interpolatedModel(self, T, G, B):
        #choose bracketing temperatures
        if not(T in self.temps):
            Tlow = max(self.temps[scipy.where(self.temps < T)])
            Thigh = min(self.temps[scipy.where(self.temps > T)])
        else:
            Tlow = Thigh = T

        #choose bracketing surface gravities
        if not(G in self.gravs):
            Glow = max(self.gravs[scipy.where(self.gravs < G)])
            Ghigh = min(self.gravs[scipy.where(self.gravs > G)])
        else:
            Glow = Ghigh = G

        #choose bracketing B-fields
        if not(B in self.bfields):
            Blow = max(self.bfields[scipy.where(self.bfields < B)])
            Bhigh  = min(self.bfields[scipy.where(self.bfields > B)])
        else:
            Blow = Bhigh = B

        #interpolate 
        y1 = self.readMOOGModel(Tlow, Glow, Blow, axis ='y')
        y2 = self.readMOOGModel(Thigh, Glow, Blow, axis ='y')
        y3 = self.readMOOGModel(Tlow, Ghigh, Blow, axis ='y')
        y4 = self.readMOOGModel(Thigh, Ghigh, Blow, axis ='y')
        y5 = self.readMOOGModel(Tlow, Glow, Bhigh, axis ='y')
        y6 = self.readMOOGModel(Thigh, Glow, Bhigh, axis ='y')
        y7 = self.readMOOGModel(Tlow, Ghigh, Bhigh, axis ='y')
        y8 = self.readMOOGModel(Thigh, Ghigh, Bhigh, axis ='y')

        new_y = numpy.zeros(len(y1))
        for i in range(len(y1)):
            if y1[i] == y2[i]:
                y12 = y1[i]
            else:
                y12 = scipy.interpolate.interp1d([Tlow, Thigh], [y1[i], y2[i]])(T)
            if y3[i] == y4[i]:
                y34 = y3[i]
            else:
                y34 = scipy.interpolate.interp1d([Tlow, Thigh], [y3[i], y4[i]])(T)
            if y5[i] == y6[i]:
                y56 = y5[i]
            else:
                y56 = scipy.interpolate.interp1d([Tlow, Thigh], [y5[i], y6[i]])(T)
            if y7[i] == y8[i]:
                y78 = y7[i]
            else:
                y78 = scipy.interpolate.interp1d([Tlow, Thigh], [y7[i], y8[i]])(T)

            if (y12==y34):
                y1234 = y12
            else:
                y1234 = scipy.interpolate.interp1d([Glow, Ghigh], [y12, y34])(G)
            if (y56 == y78):
                y5678 = y56
            else:
                y5678 = scipy.interpolate.interp1d([Glow, Ghigh], [y56, y78])(G)
            if (y1234==y5678):
                new_y[i] = y1234
            else:
                new_y[i] = scipy.interpolate.interp1d([Blow, Bhigh], [y1234, y5678])(B)

        return new_y

    def readMOOGModel(self, T, G, B, **kwargs):
        df = self.modelBaseDir+'B_'+str(B)+'kG/feat_'+str(self.features[self.currFeat]["num"])+'_MARCS_T'+str(int(T))+'G'+str(int(G))+'_R2000'
        x_sm, y_sm = SpectralTools.read_2col_spectrum(df)
        if "axis" in kwargs:
            if kwargs["axis"] == 'y':
                return y_sm
            elif kwargs["axis"] == 'x':
                return x_sm
        else:
            return x_sm, y_sm

    def traverseDescentVector(self, coordinates, x_obs, flat, z, vosd, **kwargs):
        index = -1
        x_sm = self.features[self.currFeat]["wl"]
        T=coordinates[index].T+vosd["T"]
        G=coordinates[index].G+vosd["G"]
        B=coordinates[index].B+vosd["B"]
        dx=coordinates[index].dx+vosd["dx"]
        dy=coordinates[index].dy+vosd["dy"]
        r=coordinates[index].r+vosd["r"]
        T = max(2500, T)
        T = min(6000, T)
        G = min(500, G)
        G = max(300, G)
        B = min(4.0, B)
        B = max(0.0, B)
        r = max(0.0, r)
        print "New coordinates:"
        print "T = ", T
        print "G = ", G
        print "B = ", B
        print "dx = ", dx
        print "dy = ", dy
        print "r = ", r

        y_new = self.interpolatedModel(T, G, B)
        new_wl = x_obs+dx
        overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]
        synthetic_spectrum = self.binMOOGSpectrum(y_new, x_sm, new_wl[overlap])*dy
        chisq = self.calcError(flat[overlap], (synthetic_spectrum+r)/(r+1.0), z[overlap], new_wl[overlap],
        self.features[self.currFeat]["comparePoints"])
        coordinates.append(gridPoint(chisq=chisq, T=T, G=G, B=B, dx=dx, dy=dy, r=r))
        print coordinates[index].chisq
        if "plot" in kwargs:
            fit = Gnuplot.Data(new_wl[overlap], (synthetic_spectrum+r)/(r+1.0), with_='lines')
            obs = Gnuplot.Data(new_wl[overlap], flat[overlap], with_='lines')
            kwargs["plot"].plot(fit, obs)

    def calcDescentVector(self, coordinates, x_obs, flat, z, y_old):
        x_sm = self.features[self.currFeat]["wl"]
        if (len(coordinates) == 1):
            index = -1
        else:
            index = -2
        T=coordinates[index].T
        G=coordinates[index].G
        B=coordinates[index].B
        dx=coordinates[index].dx
        dy=coordinates[index].dy
        r=coordinates[index].r
        old_chisq = coordinates[index].chisq

        dT = 250.0
        dG = 50.0
        dB = 0.5 
        d_dx = 0.000000001
        d_dy = 0.0001
        dr = 0.005

        fT=fG=fB=0.01
        fdx=fdy=fr=0.5
        
        # Calculate the dT partial derivative
        if (T < 6000-dT):
            if (T <= 3900-dT):
                dT = dT
            else:
                if (G <= 500):
                    dT = dT
                else:
                    dT = -dT
        else:
            dT = -dT
            
        y_new = self.interpolatedModel(T+dT, G, B)
        new_wl = x_obs+dx
        overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]
        synthetic_spectrum = self.binMOOGSpectrum(y_new, x_sm, new_wl[overlap])*dy
        chisq = self.calcError(flat[overlap], (synthetic_spectrum+r)/(r+1.0), z[overlap], new_wl[overlap],
        self.features[self.currFeat]["comparePoints"])
        T_step = -(chisq-old_chisq)/(fT*dT)
        coordinates.append(gridPoint(chisq=chisq, T=T+dT, G=G, B=B, dx=dx, dy=dy, r=r))
        
        # Calculate the dG partial derivative
        if (G <= 500-dG):
            dG = dG
        else:
            if ((T <= 3900) & (G < 550-dG)):
                dG = dG
            else:
                dG = -dG

        y_new = self.interpolatedModel(T, G+dG, B)
        synthetic_spectrum = self.binMOOGSpectrum(y_new, x_sm, new_wl[overlap])*dy
        chisq = self.calcError(flat[overlap], (synthetic_spectrum+r)/(r+1.0), z[overlap], new_wl[overlap],
        self.features[self.currFeat]["comparePoints"])
        G_step = -(chisq-old_chisq)/(fG*dG)
        coordinates.append(gridPoint(chisq=chisq, T=T, G=G+dG, B=B, dx=dx, dy=dy, r=r))

        # Calculate the dB partial derivative
        if (B < 4.0-dB):
            dB = dB
        else:
            dB = -dB

        y_new = self.interpolatedModel(T, G, B+dB)
        synthetic_spectrum = self.binMOOGSpectrum(y_new, x_sm, new_wl[overlap])*dy
        chisq = self.calcError(flat[overlap], (synthetic_spectrum+r)/(r+1.0), z[overlap], new_wl[overlap],
        self.features[self.currFeat]["comparePoints"])
        B_step = -(chisq-old_chisq)/(fB*dB)
        coordinates.append(gridPoint(chisq=chisq, T=T, G=G, B=B+dB, dx=dx, dy=dy, r=r))

        # Calculate the d_dx partial derivative
        new_wl = x_obs+dx+d_dx
        overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]
        synthetic_spectrum = self.binMOOGSpectrum(y_old, x_sm, new_wl[overlap])*dy
        chisq = self.calcError(flat[overlap], (synthetic_spectrum+r)/(r+1.0), z[overlap], new_wl[overlap],
        self.features[self.currFeat]["comparePoints"])
        dx_step = -(chisq-old_chisq)/(fdx*d_dx)
        dx_step = 0.0
        coordinates.append(gridPoint(chisq=chisq, T=T, G=G, B=B, dx=dx+d_dx, dy=dy, r=r))

        # Calculate the d_dy partial derivative
        new_wl = x_obs+dx
        overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]
        synthetic_spectrum = self.binMOOGSpectrum(y_old, x_sm, new_wl[overlap])*(dy+d_dy)
        chisq = self.calcError(flat[overlap], (synthetic_spectrum+r)/(r+1.0), z[overlap], new_wl[overlap],
        self.features[self.currFeat]["comparePoints"])
        dy_step = -(chisq-old_chisq)/(fdy*d_dy)
        dy_step = 0.0
        coordinates.append(gridPoint(chisq=chisq, T=T, G=G, B=B, dx=dx, dy=dy+d_dy, r=r))

        # Calculate the dr partial derivative
        synthetic_sepctrum = self.binMOOGSpectrum(y_old, x_sm, new_wl[overlap])*dy
        chisq = self.calcError(flat[overlap], (synthetic_spectrum+r+dr)/(r+dr+1.0), z[overlap], new_wl[overlap],
        self.features[self.currFeat]["comparePoints"])
        r_step = -(chisq-old_chisq)/(fr*dr)
        r_step = 0.0
        coordinates.append(gridPoint(chisq=chisq, T=T, G=G, B=B, dx=dx, dy=dy, r=r+dr))

        L = ( T_step**2.0 + G_step**2.0 + B_step**2.0 + dx_step**2.0 + dy_step**2.0 + r_step**2.0 )**0.5

        vector_of_steepest_descent = {"T":T_step*dT/(L), "G":G_step*dG/(L), "B":B_step*dB/(L),
        "dx":dx_step*d_dx/(L),"dy":dy_step*d_dy/(L), "r":r_step*dr/(L)}

        print vector_of_steepest_descent
        print L
        #raw_input()

        return vector_of_steepest_descent
	
    def fitSpectrum(self, wl, flux, error, plt):   # wl in microns
        for feat, num in zip(self.features, range(len(self.features))):
        #for feat, num in zip(self.features, range(len(self.features))):
            self.currFeat = num
            if ( min(wl) < feat["xstart"] ):
                #chisq = numpy.zeros([len(feat["TandGandB"][0]), len(self.x_offsets), len(self.y_offsets), len(self.veilings)])
                coordinates = []
                T_initial_guess = 4000.0
                G_initial_guess = 400.0
                B_initial_guess = 1.0
                dy_initial_guess = 1.0
                
		
                minchisq = 1e10
                x_window, flat, z = SEDTools.removeContinuum(wl, flux, error, feat["slope_start"], feat["slope_stop"],
                strongLines=feat["strongLines"], lineWidths=feat["lineWidths"], errors=True)
                
                T_guess = T_initial_guess
                G_guess = G_initial_guess
                B_guess = B_initial_guess
                x_sm, y_sm = self.readMOOGModel(T_guess, G_guess, B_guess)
                x_sm = x_sm/10000.0                # convert MOOG wavelengths to microns
                minx = min(x_sm)
                maxx = max(x_sm)
                self.features[self.currFeat]["wl"] = x_sm
                
                dy_guess = dy_initial_guess
                dx_guess = self.findWavelengthShift(x_window, flat, x_sm, y_sm)
                
                #Calculate the initial guess for the wavelength shift
                new_wl = x_window+dx_guess
                overlap = scipy.where( (new_wl > minx) & (new_wl < maxx) )[0]
                
                synthetic_spectrum = self.binMOOGSpectrum(y_sm, x_sm, new_wl[overlap])
                
                #Calculate the initial guess for the veiling
                #r_initial_guess = self.calcVeiling(flat[overlap], synthetic_spectrum, new_wl[overlap], feat["comparePoints"])
                r_initial_guess = 0.0
                coordinates.append(gridPoint(T=T_guess, G=G_guess, B=B_guess, dx=dx_guess, dy=dy_guess, r=r_initial_guess, chisq=self.calcError(flat[overlap], (synthetic_spectrum+r_initial_guess)/(r_initial_guess+1.0), z[overlap], new_wl[overlap], feat["comparePoints"])))
                old_chisq = coordinates[-1].chisq

                turns = 0   #Keeps track of how many changes of direction
                
                while turns < 50:
                    #Calculate the vector of steepest descent
                    vosd = self.calcDescentVector(coordinates, x_window, flat, z, y_sm)
                    #Travel along the vector of steepest descent for 1 step
                    self.traverseDescentVector(coordinates, x_window, flat, z, vosd, plot=plt)
                    while(coordinates[-1].chisq < old_chisq):   #keep continuing down vector of steepest descent
                        old_chisq = coordinates[-1].chisq
                        self.traverseDescentVector(coordinates, x_window, flat, z, vosd, plot=plt)
                    print "New direction!"
                    turns += 1

                print coordinates[-1].chisq
                print coordinates[-2].chisq
                print coordinates[0].chisq
                raw_input()
        return 'Hi'

