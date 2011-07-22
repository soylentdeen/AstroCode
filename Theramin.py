import SpectralTools
import SEDTools
import scipy
import numpy
import scipy.interpolate
import scipy.integrate
import Gnuplot
import pickle
import time


class gridPoint( object ):    # Object which contains information about a Chi-Squared grid point
    def __init__(self, TGBcoords, contLevel, veiling):
        
        self.TGB = {"T":TGBcoords["T"], "G":TGBcoords["G"], "B":TGBcoords["B"]}
        self.contLevel = {}
        self.veiling = veiling
        self.n_dims = 4
        for i in contLevel.keys():
            self.contLevel[i] = contLevel[i]
            self.n_dims += 1

    def dump(self):
        retval = 'T='+str(int(self.TGB["T"]))+', G='+str(int(self.TGB["G"]))+', B='+str(float(self.TGB["B"]))+'\n'
        for i in self.contLevel.keys():
            retval += 'Feature '+str(i)+', dy = '+str(self.contLevel[i])+'\n'

        retval += 'r_2.2 = '+str(self.veiling)+'\n'

        return retval

class spectralSynthesizer( object ):
    def __init__(self):
        feat_num = [1, 2, 3, 4, 5]
        xstart = [1.150, 1.480, 1.57, 2.170, 2.240]
        xstop = [1.2200, 1.520, 1.60, 2.230, 2.310]
        slope_start = [1.13, 1.47, 1.55, 2.100, 2.23]
        slope_stop = [1.25, 1.54, 1.62, 2.25, 2.31]
        strongLines = [[1.16, 1.168, 1.176, 1.1828, 1.1884, 1.198], [1.488, 1.503, 1.5045], [1.5745, 1.5770],
        [2.166,2.2066], [2.263, 2.267, 2.30]]
        lineWidths = [[0.005, 0.005, 0.005, 0.005, 0.005, 0.005], [0.005, 0.005, 0.01], [0.005, 0.005],[0.005, 0.005],[0.005, 0.005,0.01]]
        #comparePoints = [[[1.1816,1.1838],[1.1871,1.1900],[1.1942,1.1995],[1.2073,1.2087]], [[2.1765, 2.18], [2.1863,2.1906], [2.199, 2.2015], [2.2037, 2.210]], [[2.2525,2.2551],[2.2592, 2.2669], [2.2796, 2.2818]]]
        comparePoints = [[[1.157,1.17],[1.176,1.1995],[1.2073,1.2087]],[[1.482, 1.519]],[[1.571,1.599]], [[2.1765,2.18],[2.1863,2.1906], [2.199, 2.2015], [2.2037, 2.23]], [[2.2425,2.2551],[2.2592, 2.2669]]]

        self.modelBaseDir='/home/grad58/deen/Data/StarFormation/MOOG/zeeman/smoothed/'
        self.dataBaseDir='/home/grad58/deen/Data/StarFormation/TWA/bfields/'
        self.delta = {"T":100.0, "G":20.0, "B":0.5, "dy":0.01, "r":0.05}    # [dT, dG, dB, d_dy, dr]
        self.delta_factor = 1.0
        self.limits = {"T":[2500.0, 6000.0], "G":[300.0, 500.0], "B":[0.0,4.0], "dy":[0.98, 1.02], "r":[0.0, 10.0]}
        self.floaters = {"T":True, "G":True, "B":True, "dy":True, "r":True}
        self.features = {}

        self.calc_VeilingSED(8000.0, 2500.0, 1400.0, 62.0, 910.0)

        #for i in range(len(xstart)):
        #for i in [4]:
        for i in [0,1,3,4]:
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
            self.features[feat_num[i]] = feat

        self.temps = numpy.array(range(2500, 4000, 100)+range(4000,6250, 250))
        self.gravs = numpy.array(range(300, 600, 50))
        self.bfields = numpy.array(numpy.arange(0, 4.5, 0.5))
        self.ranges = {"T":self.temps, "G":self.gravs, "B":self.bfields}

    def calc_VeilingSED(self, Thot, Tint, Tcool, fi_fh, fc_fh):
        wave = numpy.arange(1.0, 2.5, 0.01)
        bm = numpy.argsort(abs(wave-2.2))[0]
        Bhot = SpectralTools.blackBody(wl = wave/10000.0, T=Thot, outUnits='Energy')
        Bint = SpectralTools.blackBody(wl = wave/10000.0, T=Tint, outUnits='Energy')*fi_fh
        Bcool = SpectralTools.blackBody(wl = wave/10000.0, T=Tcool, outUnits='Energy')*fc_fh
        composite = Bhot+Bint+Bcool
        composite /= composite[bm]
        self.veiling_SED = scipy.interpolate.interp1d(wave, composite)

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

    def calcError(self, y1, y2, z, x, comparePoints, **kwargs):
        error = 0.0
        if ("plt" in kwargs):
            obs = Gnuplot.Data(x, y1, with_='lines')
            new = Gnuplot.Data(x, y2, with_='lines')
            kwargs["plt"].plot(obs, new)
        bm = []
        for region in comparePoints:
            bm.extend(scipy.where( (x > region[0]) & (x < region[1]) )[0])
                
        for dat in zip(y1[bm], y2[bm], z[bm]):
            error += ((dat[0]-dat[1])/dat[2])**2
            
        return error/len(y1[bm])


    def findWavelengthShift(self, x_window, flat, x_sm, y_sm):
        if self.currFeat == 2:
            y_sm *= 0.95
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
            
        self.features[self.currFeat]["x_offset"] = offset_computed

    def calcVeiling(self, y_obs, y_calc, x, comparePoints):
        bm = []
        for region in comparePoints:
            bm.extend(scipy.where( (x > region[0]) & (x <region[1]) )[0])
            
        rk_guess = [0.01]
        ff = lambda p, y_c: (y_c+p[0])/(abs(p[0])+1.0)
        ef = lambda p, y_c, y_o: ff(p, y_c)-y_o
        veil, success = scipy.optimize.leastsq(ef, rk_guess, args = (y_calc[bm], y_obs[bm]))
        
        '''
        plt = Gnuplot.Gnuplot()
        a = Gnuplot.Data(x[bm], y_obs[bm], with_='lines')
        b = Gnuplot.Data(x[bm], y_calc[bm], with_='lines')
        c = Gnuplot.Data(x[bm], (y_calc[bm]+veil[0])/(1.0+veil[0]), with_='lines')
        plt.plot(a, b, c)
        raw_input()
        '''

        return veil[0]

    def interpolatedModel(self, T, G, B):
        #choose bracketing temperatures
        if not(T in self.temps):
            Tlow = max(self.temps[scipy.where(self.temps <= T)])
            Thigh = min(self.temps[scipy.where(self.temps >= T)])
        else:
            Tlow = Thigh = T

        #choose bracketing surface gravities
        if not(G in self.gravs):
            Glow = max(self.gravs[scipy.where(self.gravs <= G)])
            Ghigh = min(self.gravs[scipy.where(self.gravs >= G)])
        else:
            Glow = Ghigh = G

        #choose bracketing B-fields
        if not(B in self.bfields):
            Blow = max(self.bfields[scipy.where(self.bfields <= B)])
            Bhigh  = min(self.bfields[scipy.where(self.bfields >= B)])
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
        df = self.modelBaseDir+'B_'+str(B)+'kG/f'+str(self.features[self.currFeat]["num"])+'_MARCS_T'+str(int(T))+'G'+str(int(G))+'_R2000'
        x_sm, y_sm = SpectralTools.read_2col_spectrum(df)
        if "axis" in kwargs:
            if kwargs["axis"] == 'y':
                return y_sm
            elif kwargs["axis"] == 'x':
                return x_sm
        else:
            return x_sm, y_sm

    def computeS(self, coordinates, **kwargs):
        retval = 0.0
        for num in coordinates.contLevel.keys():
            self.currFeat = num
            y_new = self.interpolatedModel(coordinates.TGB["T"], coordinates.TGB["G"], coordinates.TGB["B"])
            x_sm = self.features[self.currFeat]["wl"]

            new_wl = self.x_window[self.currFeat] + self.features[self.currFeat]["x_offset"]
            overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]

            synthetic_spectrum = self.binMOOGSpectrum(y_new, x_sm, new_wl[overlap])*coordinates.contLevel[num]
            excess = self.compute_Excess(new_wl[overlap], coordinates.TGB["T"], coordinates.veiling)
            #excess = self.veiling_SED(new_wl[overlap])*veiling
            veiled = (synthetic_spectrum+excess)/(excess+1.0)
            #'''
            if "plot" in kwargs:
                obs = Gnuplot.Data(new_wl[overlap], self.flat[num][overlap], with_='lines')
                sim = Gnuplot.Data(new_wl[overlap], synthetic_spectrum, with_='lines')
                veil = Gnuplot.Data(new_wl[overlap], veiled, with_='lines')
                kwargs["plot"].plot(obs, sim, veil)
                time.sleep(2.0)
            #'''
            retval += self.calcError(self.flat[self.currFeat][overlap],veiled,self.z[self.currFeat][overlap], new_wl[overlap], self.features[self.currFeat]["comparePoints"])

        return retval
    
    def compute_Excess(self, wl, Teff, veiling):
        excess_BB = self.veiling_SED(wl)
        star_BB = SpectralTools.blackBody(wl = wl/10000.0, T=Teff, outUnits="Energy")
        zp = SpectralTools.blackBody(wl=2.2/10000.0, T=Teff, outUnits="Energy")
        star_BB /= zp
        excess = excess_BB/star_BB*veiling
        
        return excess
    
    def computeDeriv(self, coords, index):
        coords[index] += self.delta[index]
        S1 = self.computeS(coords)

        coords[index] -= 2.0*self.delta[index]
        S2 = self.computeS(coords)

        coords[index] += self.delta[index]

        return (S1-S2)/(2*self.delta[index])

    def compute2ndDeriv(self, coords, i, j):
        if( i == j ):
            ifactor = 1.0
            while (((coords[i] - self.delta[i]*ifactor) < self.limits[i][0]) | ((coords[i]+self.delta[i]*ifactor) >
            self.limits[i][1])):
                ifactor *= 0.8
            S2 = self.computeS(coords)
            coords[i] += self.delta[i]*ifactor
            S1 = self.computeS(coords)
            coords[i] -= 2*self.delta[i]*ifactor
            S3 = self.computeS(coords)
            coords[i] += self.delta[i]*ifactor
            return (S1-2*S2+S3)/(self.delta[i]*self.delta[i]*ifactor*ifactor)
        else:
            ifactor = 1.0
            while (((coords[i] - self.delta[i]*ifactor) < self.limits[i][0]) | ((coords[i]+self.delta[i]*ifactor) >
            self.limits[i][1])):
                ifactor *= 0.8
            jfactor = 1.0
            while (((coords[j] - self.delta[j]*jfactor) < self.limits[j][0]) | ((coords[j]+self.delta[j]*jfactor) >
            self.limits[j][1])):
                jfactor *= 0.8
            coords[i] += self.delta[i]*ifactor
            coords[j] += self.delta[j]*jfactor
            S1 = self.computeS(coords)

            coords[i] -= 2.0*self.delta[i]*ifactor
            S2 = self.computeS(coords)

            coords[j] -= 2.0*self.delta[j]*jfactor
            S4 = self.computeS(coords)

            coords[i] += self.delta[i]*ifactor
            S3 = self.computeS(coords)

            coords[j] += self.delta[j]*jfactor

            return (S1 - S2 - S3 + S4)/(4*(self.delta[i]*ifactor)*(self.delta[j]*jfactor))

    def computeGradient(self, coords):
        G = numpy.zeros(len(coords))

        for i in range(len(coords)):
            G[i] = self.computeDeriv(coords, i)

        return numpy.matrix(G)

    def computeHessian(self, coords):
        new_coords = coords
        H = numpy.zeros([len(coords),len(coords)])

        for i in range(len(coords)):
            for j in range(len(coords)):
                H[i,j] = self.compute2ndDeriv(new_coords, i, j)
                print i, j, H[i,j]

        return numpy.matrix(H)

    def marquardt(self, chisq):
        self.alpha = 0.01
        old_chisq = chisq[0][0]
        old_coordinates = chisq[0][1]
        
        while( old_chisq > 10):
            print old_chisq
            print old_coordinates
            G = self.computeGradient(old_coordinates)
            H = self.computeHessian(old_coordinates)
            H_prime = H*numpy.matrix(numpy.eye(len(old_coordinates), len(old_coordinates))*(1.0+self.alpha))
            
            vect = H_prime.I*G.transpose()
            new_coordinates = old_coordinates+vect.transpose()
            old_chisq = self.computeS(numpy.array(new_coordinates)[0])
            old_coordinates = numpy.array(new_coordinates)[0]

    def check_bounds(self, coords):
        retval = []
        if ( coords[0] >= 2500.0 ):
            if ( coords[0] <= 6000.0):
                retval.append(coords[0])
            else:
                retval.append(6000.0)
        else:
            retval.append(2500.0)

        if ( coords[1] >= 300.0):
            if (coords[1] <= 500.0):
                retval.append(coords[1])
            else:
                retval.append(500.0)
        else:
            retval.append(300.0)

        if (coords[2] >= 0.0):
            if (coords[2] <= 4.0):
                retval.append(coords[2])
            else:
                retval.append(4.0)
        else:
            retval.append(0.0)

        retval.append(coords[3])
        retval.append(coords[4])
        if (coords[5] > 0.0):
            retval.append(coords[5])
        else:
            retval.append(0.0)

        return retval

    def gridSearch(self, **kwargs):
        Tval = []
        Gval = []
        Bval = []
        veiling = []
        S = []

        outfile = kwargs["outfile"]
        
        if kwargs["mode"] == 'FINAL':
            infile = open(outfile)
            for line in infile.readlines()[1:]:
                l = line.split()
                Tval.append(float(l[0]))
                Gval.append(float(l[1]))
                Bval.append(float(l[2]))
                veiling.append(float(l[4]))
                S.append(float(l[5]))
            x_offset = float(l[3])
            infile.close()
        elif kwargs["mode"] == 'PREP':
            out = open(outfile, 'w')
            x_sm = self.features[self.currFeat]["wl"]
            x_offset = self.features[self.currFeat]["x_offset"]
            new_wl = self.x_window+x_offset
            out.write('#%10s %10s %10s %10s %10s %10s\n' % ('Temp', 'Grav', 'Bfield', 'X offset', 'Veiling', 'Chi-Square'))

            overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]

            for B in self.bfields:
                for T in self.temps:
                    print B, T
                    for G in self.gravs:
                        flag = False
                        if (T < 3900.0):
                            flag = True
                        elif (G < 500):
                            flag = True
                        if flag == True:
                            y_sm = self.readMOOGModel(T, G, B, axis ='y')
                            synthetic_spectrum = self.binMOOGSpectrum(y_sm, x_sm, new_wl[overlap])
                            #Calculate the initial guess for the veiling
                            veiling.append(self.calcVeiling(self.flat[overlap], synthetic_spectrum,
                            new_wl[overlap],self.features[self.currFeat]["comparePoints"]))
                            S.append(self.calcError(self.flat[overlap], (synthetic_spectrum+veiling[-1])/(veiling[-1]+1.0),
                            self.z[overlap], new_wl[overlap], self.features[self.currFeat]["comparePoints"],
                            plt = kwargs["plot"]))
                            Tval.append(T)
                            Gval.append(G)
                            Bval.append(B)
                            out.write('%10.3f %10.3f %10.3f %10.4e %10.4e %10.3e\n' % (T,G,B,x_offset,veiling[-1], S[-1]) )

            out.close()
            
        order = numpy.argsort(S)

        Tval = numpy.array(Tval)
        Gval = numpy.array(Gval)
        Bval = numpy.array(Bval)
        S = numpy.array(S)
        veiling = numpy.array(veiling)
        self.features[self.currFeat]["x_offset"] = x_offset

        '''
        for B in self.bfields:
            for G in self.gravs:
                bm = scipy.where( (Gval==G) & (Bval==B) )[0]
                a = Gnuplot.Data(Tval[bm], S[bm], with_='lines')
                kwargs["plot"].plt('set title "B='+str(B)+', G ='+str(G)+'"')
                kwargs["plot"].plt.plot(a)
                raw_input()'''

        #initial_guess = [numpy.mean(Tval[order[0:20]]), numpy.mean(Gval[order[0:20]]), numpy.mean(Bval[order[0:20]]), x_offset, 1.00,0.00]
        initial_guess = {"T":numpy.mean(Tval[order[0:20]]), "G":numpy.mean(Gval[order[0:20]]),"B":numpy.mean(Bval[order[0:20]])}
        return initial_guess

    def findCentroid(self, coords):
        T = 0.0
        G = 0.0
        B = 0.0
        n_coords = float(len(coords))
        n_feat = len(coords[0].contLevel.keys())
        dy = {key:0.0 for key in coords[0].contLevel.keys()}
        r = 0.0
        for coord in coords:
            T += coord.TGB["T"]/n_coords
            G += coord.TGB["G"]/n_coords
            B += coord.TGB["B"]/n_coords
            r += coord.veiling/n_coords
            for i in coord.contLevel.keys():
                dy[i] += coord.contLevel[i]/n_coords

        TGB = {"T":T, "G":G, "B":B}
        cL = {}
        for i in coords[0].contLevel.keys():
            cL[i] = dy[i]

        retval = gridPoint(TGB, cL, r)
        return retval
            
    def reflect(self, centroid, coord, **kwargs):
        if kwargs["trial"] == 1:
            new_T = 2.0*centroid.TGB["T"] - coord.TGB["T"]
            new_G = 2.0*centroid.TGB["G"] - coord.TGB["G"]
            new_B = 2.0*centroid.TGB["B"] - coord.TGB["B"]
            new_dy = {key:0.0 for key in coord.contLevel.keys()}
            new_r = 2.0*centroid.veiling - coord.veiling
            for i in coord.contLevel.keys():
                new_dy[i] = 2.0*centroid.contLevel[i] - coord.contLevel[i]

            TGB = {"T":new_T, "G":new_G, "B":new_B}
            cL = {key:0.0 for key in coord.contLevel.keys()}
            for i in coord.contLevel.keys():
                cL[i] = new_dy[i]

            retval = gridPoint(TGB, cL, new_r)

        elif kwargs["trial"] == 2:
            new_T = 3.0*centroid.TGB["T"] - 2.0*coord.TGB["T"]
            new_G = 3.0*centroid.TGB["G"] - 2.0*coord.TGB["G"]
            new_B = 3.0*centroid.TGB["B"] - 2.0*coord.TGB["B"]
            n_feat = len(coord.contLevel)
            new_dy = {key:0.0 for key in coord.contLevel.keys()}
            new_r = 3.0*centroid.veiling - 2.0*coord.veiling
            for i in coord.contLevel.keys():
                new_dy[i] = 3.0*centroid.contLevel[i] - 2.0*coord.contLevel[i]

            TGB = {"T":new_T, "G":new_G, "B":new_B}
            cL = {key:0.0 for key in coord.contLevel.keys()}
            for i in coord.contLevel.keys():
                cL[i] = new_dy[i]
            retval = gridPoint(TGB, cL, new_r)

        elif kwargs["trial"] == 3:
            new_T = centroid.TGB["T"] - (centroid.TGB["T"] - coord.TGB["T"])/2.0
            new_G = centroid.TGB["G"] - (centroid.TGB["G"]-coord.TGB["G"])/2.0
            new_B = centroid.TGB["B"] - (centroid.TGB["B"]-coord.TGB["B"])/2.0
            n_feat = len(coord.contLevel)
            new_dy = {key:0.0 for key in coords.contLevel.keys()}
            new_r = centroid.veiling - (centroid.veiling-coord.veiling)/2.0
            for i in coord.contLevel.keys():
                new_dy[i] = centroid.contLevel[i]-(centroid.contLevel[i]-coord.contLevel[i])/2.0

            TGB = {"T":new_T, "G":new_G, "B":new_B}
            cL = {key:0.0 for key in coords.contLevel.keys()}
            for i in coord.contLevel.keys():
                cL[i] = new_dy[i]
            retval = gridPoint(TGB, cL, new_r)
            
        return retval

    def contract(self, centroid, coords):
        for i in range(len(coords)):
            newcoord = self.reflect(centroid, coords[i], trial=3)
            coords[i] = newcoord

        return coords

    def simplex(self, init_guess, plt):
        #simplex_coords = [init_guess]
        #simplex_values = [self.computeS(init_guess)]

        coords = []
        Svalues = []
        for key in self.floaters.keys():
            if self.floaters[key] == True:
                new_TGBcoords = {"T":init_guess["T"], "G":init_guess["G"], "B":init_guess["B"]}
                if key in new_TGBcoords:
                    if (new_TGBcoords[key] + self.delta[key]) < self.limits[key][1]:
                        new_TGBcoords[key] += self.delta[key]
                    else:
                        new_TGBcoords[key] -= self.delta[key]
                    contLevels = {}
                    for num in self.x_window.keys():
                        contLevels[num] = 1.0
                    veil = init_guess["r"]
                    simplexPoint = gridPoint(new_TGBcoords, contLevels, veil)
                    coords.append(simplexPoint)
                    Svalues.append(self.computeS(simplexPoint))
                elif key == 'dy':
                    for num in self.x_window.keys():
                        contLevels = {}
                        for i in self.x_window.keys():
                            contLevels[i] = 1.0
                        if (contLevels[num] + self.delta["dy"]) < self.limits["dy"][1]:
                            contLevels[num] += self.delta["dy"]
                        else:
                            contLevels[num] -= self.delta["dy"]
                        veil = init_guess["r"]
                        simplexPoint = gridPoint(new_TGBcoords, contLevels, veil)
                        coords.append(simplexPoint)
                        Svalues.append(self.computeS(simplexPoint))
                elif key == 'r':
                    contLevels = {}
                    for i in self.x_window.keys():
                        contLevels[i] = 1.0
                    veil = init_guess["r"]
                    if (veil + self.delta["r"]) < self.limits["r"][1]:
                        veil += self.delta["r"]
                    else:
                        veil -= self.delta["r"]
                    simplexPoint = gridPoint(new_TGBcoords, contLevels, veil)
                    coords.append(simplexPoint)
                    Svalues.append(self.computeS(simplexPoint))

        n_contractions = 0

        print len(coords)
        while (numpy.std(Svalues) > 1.5):
            print numpy.mean(Svalues), numpy.std(Svalues)
            order = numpy.argsort(Svalues)
            centroid = self.findCentroid(coords)
            print centroid.dump()

            junk = self.computeS(centroid, plot=plt)
            # Reflect about centroid
            trial_1 = self.reflect(centroid, coords[order[-1]], trial=1)
            S1 = self.computeS(trial_1)
            
            if (S1 > Svalues[order[-1]]):    # Try longer path along distance
                trial_2 = self.reflect(centroid, coords[order[-1]], trial=2)
                S2 = self.computeS(trial_2)
                if (S2 > Svalues[order[-1]]):    # Try half the distance
                    trial_3 = self.reflect(centroid, coords[order[-1]], trial=3)
                    S3 = self.computeS(trial_3)
                    if (S3 > Svalues[order[-1]]):     # Shrink?
                        #self.delta_factor *= 2.0
                        if n_contractions <= 2:
                            coords = self.contract(centroid, coords)
                            #print 'Contracted!'
                            n_contractions += 1
                        else:
                            break
                    else:
                        coords[order[-1]] = trial_3
                        Svalues[order[-1]] = S3
                        #print 'Trial 3'
                else:
                    coords[order[-1]] = trial_2
                    Svalues[order[-1]] = S2
                    #print 'Trial 2'
            else:
                coords[order[-1]] = trial_1
                Svalues[order[-1]] = S1
                #print 'Trial 1'
        
        retval = self.findCentroid(coords)
        junk = self.computeS(retval, plot=plt)
        print numpy.mean(Svalues), numpy.std(Svalues)

        '''
        new_wl = self.x_window + centroid[3]
        overlap = scipy.where( (new_wl > min(self.features[self.currFeat]["wl"])) & (new_wl < max(self.features[self.currFeat]["wl"])) )[0]
        x = new_wl[overlap]
        bm = []
        for region in self.features[self.currFeat]["comparePoints"]:
            bm.extend(scipy.where( (x > region[0]) & (x < region[1]) )[0])
        H = self.computeHessian(centroid)
        C = 2.0*H.I
        covar = min(best_values)*C*len(bm)/(len(bm)-len(best_values))

        print best_coords
        print best_values
        print covar
        raw_input()
        '''
        print retval.dump()
        return retval

    def fitSpectrum(self, wl, flux, error, plt, **kwargs):
        outfile = kwargs["outfile"]
        # Gets the initial guess coordinates for both features
        guess_coords = {}
        self.x_window = {}
        self.flat = {}
        self.z = {}
        for num in self.features.keys():
            feat = self.features[num]
            self.currFeat = num
            if (min(wl) < feat["xstart"]):
                x_window, flat, z = SEDTools.removeContinuum(wl, flux, error, feat["slope_start"], feat["slope_stop"],
                strongLines=feat["strongLines"], lineWidths=feat["lineWidths"], errors=True)

                self.x_window[num] = x_window
                self.flat[num] = flat
                self.z[num] = z

                x_sm, y_sm = self.readMOOGModel(4000.0, 400.0, 0.0)
                x_sm = x_sm/10000.0                # convert MOOG wavelengths to microns
                self.features[self.currFeat]["wl"] = x_sm
                
                kwargs["outfile"]=self.dataBaseDir+outfile+'_feat_'+str(self.features[self.currFeat]["num"])+'.dat'
                guess_coords[num] = self.gridSearch(**kwargs)
                
        initial_guess = {}
        initial_guess["T"] = numpy.mean([guess_coords[coord]["T"] for coord in guess_coords])
        if (2 in guess_coords.keys() ):
            initial_guess["G"] = min(guess_coords[2]["G"], 500)
        else:
            initial_guess["G"] = numpy.mean([guess_coords[coord]["G"] for coord in guess_coords])
        initial_guess["B"] = numpy.mean([guess_coords[coord]["B"] for coord in guess_coords])
        
        # Calculate an initial guess for the veiling at 2.2 microns (feature 4)
        x_sm = self.features[4]["wl"]
        self.currFeat = 4
        y_new = self.interpolatedModel(initial_guess["T"], initial_guess["G"], initial_guess["B"])

        new_wl = self.x_window[4] + self.features[4]["x_offset"]
        overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]

        synthetic_spectrum = self.binMOOGSpectrum(y_new, x_sm, new_wl[overlap])
        initial_guess["r"] = self.calcVeiling(self.flat[4][overlap], synthetic_spectrum, new_wl[overlap],self.features[4]["comparePoints"])
        
        print "Guesses from Grid Search :", guess_coords
        print "Initial Guess :", initial_guess
        #retval.append(initial_guess)
        best_coords = self.simplex(initial_guess, plt)
        print 'Best Fit Coordinates :', best_coords

        return best_coords

    def prelimSearch(self, wl, flux, error, plt, **kwargs):   # wl in microns
        retval = []
        outfile = kwargs["outfile"]
        for feat, num in zip(self.features, range(len(self.features))):
        #for feat, num in zip(self.features, range(len(self.features))):
            self.delta_factor = 1.0
            self.currFeat = num
            if ( min(wl) < feat["xstart"] ):
                #chisq = numpy.zeros([len(feat["TandGandB"][0]), len(self.x_offsets), len(self.y_offsets), len(self.veilings)])
                x_window, flat, z = SEDTools.removeContinuum(wl, flux, error, feat["slope_start"], feat["slope_stop"],strongLines=feat["strongLines"], lineWidths=feat["lineWidths"], errors=True)
                
                self.x_window = x_window
                self.flat = flat
                self.z = z
                
                if kwargs["mode"] == 'PREP':
                    coordinates = []
                    T_initial_guess = 4000.0
                    G_initial_guess = 400.0
                    B_initial_guess = 1.0
                    dy_initial_guess = 1.0
                    
                    T_guess = T_initial_guess
                    G_guess = G_initial_guess
                    B_guess = B_initial_guess
                    x_sm, y_sm = self.readMOOGModel(T_guess, G_guess, B_guess)
                    x_sm = x_sm/10000.0                # convert MOOG wavelengths to microns
                    minx = min(x_sm)
                    maxx = max(x_sm)
                    self.features[self.currFeat]["wl"] = x_sm
                    
                    
                    dy_guess = dy_initial_guess
                    self.findWavelengthShift(x_window, flat, x_sm, y_sm)
                    dx_guess = self.features[self.currFeat]["x_offset"]
                    
                    #Calculate the initial guess for the wavelength shift
                    new_wl = x_window+dx_guess
                    overlap = scipy.where( (new_wl > minx) & (new_wl < maxx) )[0]
                    
                    synthetic_spectrum = self.binMOOGSpectrum(y_sm, x_sm, new_wl[overlap])

                    
                    #Calculate the initial guess for the veiling
                    #r_initial_guess = self.calcVeiling(flat[overlap], synthetic_spectrum, new_wl[overlap], feat["comparePoints"])
                    kwargs["outfile"]=self.dataBaseDir+outfile+'_feat_'+str(self.features[self.currFeat]["num"])+'.dat'
                    guess_coords = self.gridSearch(plot=plt,**kwargs)

                    retval.append([0])
        return retval

