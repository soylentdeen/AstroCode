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

class spectralSynthesizerCoarse( object ):   # low resolution

    def __init__(self):
        feat_num = [12, 57, 8]
        xstart = [1.190, 2.170, 2.260]
        xstop = [1.2100, 2.2112, 2.268]
        slope_start = [1.15, 2.100, 2.23]
        slope_stop = [1.25, 2.25, 2.29]
        continuum = [[1.2015, 1.2025], [2.1922, 2.1975], [2.258, 2.26]]
        strongLines = [[2.166, 2.2066], [2.166, 2.2066], [2.263, 2.267]]
        lineWidths = [[0.005, 0.005],[0.005, 0.005], [0.005, 0.005]]
        comparePoints = [[[1.1816,1.1838],[1.1871,1.1900],[1.1942,1.1995],[1.2073,1.2087]], [[2.1765, 2.18], [2.1863,
        2.1906], [2.199, 2.2015], [2.2037, 2.210]], [[2.2525,2.2551],[2.2592, 2.2669], [2.2796, 2.2818]]]

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
            feat["datadir"] = '/home/deen/Data/StarFormation/MOOG/features/feat_'+str(feat["num"])+'/MARCS_output/'
            feat["models"], feat["wl"], feat["TandG"] = self.getMARCSModelsCoarse(feat["num"])
            feat["comparePoints"] = comparePoints[i]
            self.features.append(feat)

        temps = numpy.array([])
        gravs = numpy.array([])

        for feat in self.features:
            temps = numpy.append(temps, feat["TandG"][0])
            gravs = numpy.append(gravs, feat["TandG"][1])

        self.temps = numpy.unique(temps)
        self.gravs = numpy.unique(gravs)
        self.x_offsets = numpy.linspace(0, 0, num = 1)  #wavelength offsets in angstroms
        self.y_offsets = numpy.linspace(0.99, 1.01, num=3)   #continuum offsets in % of continuum
        self.veilings = numpy.linspace(0.0, 1.0, num=5)

        self.temps.sort()
        self.gravs.sort()

        #self.coarse_t = numpy.arange(3900, 5400, 100)
        #self.coarse_t = self.temps[numpy.arange(0, len(self.temps), 2)]
        #self.coarse_g = numpy.arange(300, 560, 20)
        #self.coarse_g = self.gravs[numpy.arange(0, len(self.gravs), 2)]
        #self.coarse_xoff = self.x_offsets#[numpy.arange(0, len(self.x_offsets), 3)]
        #self.coarse_yoff = numpy.array([1.000])
        #self.coarse_yoff = self.y_offsets#[numpy.arange(0, len(self.y_offsets), 4)]

    def getMARCSModelsCoarse(self, fnum):
        filename = "/home/deen/Code/python/StarFormation/MOOG/fit/feature_"+str(fnum)+"_models_coarse.pkl"

        data = pickle.load(open(filename) )
        models = data[0][0]
        wl = data[0][1]
        TandG = numpy.array(data[0][2])

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


    def fitSpectrum(self, wl, flux, error, plt):   # wl in microns
        for feat in self.features:
            if ( min(wl) < feat["xstart"] ):
                chisq = numpy.zeros([len(feat["TandG"][0]), len(self.x_offsets), len(self.y_offsets), len(self.veilings)])
                minchisq = 1e10
                x_window, flat, z = SEDTools.removeContinuum(wl, flux, error, feat["slope_start"], feat["slope_stop"],
                strongLines=feat["strongLines"], lineWidths=feat["lineWidths"], errors=True)
                x_sm = feat["wl"]/10000.0     # convert MOOG wavelengths to microns
                minx = min(x_sm)
                maxx = max(x_sm)

                TandG_bm = scipy.where( (feat["TandG"][0] == 4000.0) & (feat["TandG"][1] == 400.0) )[0]
                y_sm = feat["models"][TandG_bm]
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
                nLags = xcorr-(len(orig_bm)-1)
                offset_computed = nLags*(feature_x[0]-feature_x[1])
                if abs(offset_computed) > 20:
                    offset_computed = 0


                for wl_offset in self.x_offsets:
                    print wl_offset
                    new_wl = x_window+(wl_offset/10000.0+offset_computed)       #microns
                    overlap = scipy.where( (new_wl > minx) & (new_wl < maxx))[0]
                    xoff_bm = scipy.where(self.x_offsets == wl_offset)[0]
                    for fl_offset in self.y_offsets:
                        new_fl = flat*fl_offset
                        print fl_offset
                        yoff_bm = scipy.where(self.y_offsets == fl_offset)[0]
                        for T in self.temps:
                             for G in self.gravs:
                                 TandG_bm = scipy.where( (feat["TandG"][0] == T) & (feat["TandG"][1] == G) )[0]
                                 if len(TandG_bm) > 0:
                                     for r in range(len(self.veilings)):
                                         y_sm = (feat["models"][TandG_bm]+self.veilings[r])/(self.veilings[r]+1.0)
                                         synthetic_spectrum = self.binMOOGSpectrum(y_sm, x_sm, new_wl[overlap])
                                         chisq[TandG_bm, xoff_bm, yoff_bm,r] = self.calcError(new_fl[overlap], synthetic_spectrum,
                                         z[overlap], new_wl[overlap], feat["comparePoints"])
                                         if chisq[TandG_bm, xoff_bm, yoff_bm, r] < minchisq:
                                             minchisq = chisq[TandG_bm, xoff_bm, yoff_bm, r]
                                             print minchisq, T, G, fl_offset, wl_offset, self.veilings[r]
                                             obs = Gnuplot.Data(new_wl[overlap], new_fl[overlap], with_='lines')
                                             syn = Gnuplot.Data(new_wl[overlap], synthetic_spectrum, with_='lines')
                                             plt.plot(obs, syn)

                chisq = numpy.array(chisq)
                order = numpy.argsort(chisq)
            '''
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
            '''

