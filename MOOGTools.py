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

class spectralSynthesizer( object ):

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

