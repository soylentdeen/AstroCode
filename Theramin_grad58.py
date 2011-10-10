import SpectralTools
import SEDTools
import scipy
import numpy
import scipy.interpolate
import scipy.integrate
import scipy.optimize
import Gnuplot
import pickle
import time
import copy
import os
import matplotlib.pyplot as pyplot
import matplotlib
import matplotlib.colors


class gridPoint( object ):    # Object which contains information about a Chi-Squared grid point
    def __init__(self, TGBcoords, contLevel, veiling):
        self.limits = {"T":[2500.0, 6000.0], "G":[300.0, 500.0], "B":[0.0,4.0], "r":[0.000025, 10.0]}
        self.coords = {"T":TGBcoords["T"], "G":TGBcoords["G"], "B":TGBcoords["B"], "r":veiling}
        for i in contLevel.keys():
            self.coords[i] = contLevel[i]
            self.limits[i] = [0.5, 1.5]
        self.n_dims = len(self.coords)
        self.S = 0.0

    def addChiSquared(self, S):    # Adds the Chi Squared associated with this point
        self.S = S

    def dump(self):                # Prints out the contents of the gridPoint in text format
        retval = ''
        for i in self.coords.keys():
            retval += 'Feature '+str(i)+', dy = '+str(self.coords[i])+'\n'
        retval += 'Chi Squared : '+str(self.S)+'\n'
        return retval

    def difference(self, other):
        for i in self.coords.keys():
            retval += 'Difference in Feature '+str(i)+' = '+str(self.coords[i]-other.coords[i])+'\n'

        retval += 'Difference in Chi-Squared : '+ str(self.S - other.S) + '\n'
        return retval
    
    def checkLimits(self):
        for key in self.limits.keys():
            if ( self.coords[key] >= self.limits[key][0] ):
                if ( self.coords[key] <= self.limits[key][1]):
                    pass
                else:
                    self.coords[key] =self.limits[key][1]
            else:
                self.coords[key] = self.limits[key][0]

    def __lt__(self, other):
        return self.S < other.S

class spectralSynthesizer( object ):
    def __init__(self):
        feat_num = ['alpha', 'bravo', '-3', 'charlie', 'delta']
        xstart = [1.150, 1.480, 1.57, 2.170, 2.240]
        xstop = [1.2200, 1.520, 1.60, 2.230, 2.310]
        slope_start = [1.15, 1.47, 1.55, 2.100, 2.23]
        slope_stop = [1.25, 1.54, 1.62, 2.25, 2.31]
        strongLines = [[1.16, 1.168, 1.176, 1.1828, 1.1884, 1.198], [1.488, 1.503, 1.5045], [1.5745, 1.5770],
        [2.166,2.2066], [2.263, 2.267, 2.30]]
        lineWidths = [[0.0035, 0.0035, 0.0035, 0.0025, 0.0025, 0.0025], [0.005, 0.005, 0.01], [0.005, 0.005],[0.005, 0.005],[0.005, 0.005,0.01]]
        #comparePoints = [[[1.1816,1.1838],[1.1871,1.1900],[1.1942,1.1995],[1.2073,1.2087]], [[2.1765, 2.18], [2.1863,2.1906], [2.199, 2.2015], [2.2037, 2.210]], [[2.2525,2.2551],[2.2592, 2.2669], [2.2796, 2.2818]]]
        comparePoints = [[[1.157,1.17],[1.18,1.1995],[1.2073,1.212]],[[1.482, 1.519]],[[1.571,1.599]],
        [[2.1765,2.18],[2.1863,2.2015], [2.203, 2.23]],[[2.2425, 2.31]]]#[[2.2425,2.258],[2.270,2.287]]]#[2.2425,2.31]]] 
        #[[2.1765,2.18],[2.1863,2.2015], [2.203, 2.23]], [[2.2425,2.258]]]
        continuumPoints = [[[1.165,1.168],[1.171,1.174],[1.19,1.195],[1.211, 1.22]],[[1.49,1.50],[1.508,1.52]],[[1.57, 1.60]],[[2.192,2.196],[2.211,2.218]],[[2.24, 2.258],[2.27, 2.277],[2.86,2.291]]]

        self.modelBaseDir='/home/grad58/deen/Data/StarFormation/MOOG/zeeman/stokes_smoothed/'
        self.dataBaseDir='/home/grad58/deen/Data/StarFormation/bfields/stokes/'
        self.delta = {"T":100.0, "G":50.0, "B":0.25, "dy":0.00025, "r":0.1}    # [dT, dG, dB, d_dy, dr]
        #self.limits = {"T":[2500.0, 6000.0], "G":[300.0, 500.0], "B":[0.0,4.0], "dy":[0.95, 1.05], "r":[0.0, 10.0]}
        self.floaters = {"T":True, "G":True, "B":True, "dy":True, "r":True}
        self.features = {}
        self.convergencePoints = []

        #self.calc_VeilingSED(8000.0, 5000.0, 1400.0, 30.0, 6100.0)
        self.calc_VeilingSED(8000.0, 2500.0, 1400.0, 62.0, 910.0)

        #for i in range(len(xstart)):
        #for i in [4]:
        for i in [0,1,3, 4]:
            feat = dict()
            feat["num"] = feat_num[i]
            feat["xstart"] = xstart[i]
            feat["xstop"] = xstop[i]
            feat["slope_start"] = slope_start[i]
            feat["slope_stop"] = slope_stop[i]
            feat["strongLines"] = strongLines[i]
            feat["lineWidths"] = lineWidths[i]
            feat["comparePoints"] = comparePoints[i]
            feat["continuumPoints"] = continuumPoints[i]
            self.features[feat_num[i]] = feat

        temps = numpy.array(range(3000, 4100, 100))#+range(4000,5250, 250))
        gravs = numpy.array(range(300, 550, 50))
        bfields = numpy.array(numpy.arange(0, 4.5, 0.5))
        self.ranges = {"T":numpy.array(range(2500, 4000, 100)+range(4000, 6250, 250)), "G":numpy.array(range(300,
        550, 50)), "B":numpy.array(numpy.arange(0, 4.5, 0.5))}
        self.coarseRanges = {"T":numpy.array(range(2600, 4000, 200)+range(4000, 6500, 500)), "G":numpy.array(range(300,
        600, 100)), "B":numpy.array(numpy.arange(0, 4.5, 1.0))}

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

    def calcError(self, y1, y2, z, x, comparePoints, **kwargs):    #Workhorse of the Chi-Squared Calculation routine
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
            
        #return error/len(y1[bm])            # uncomment this to return reduced Chi-Squared
        return error

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

    def continuumTest(self, guess, coord, keys):
        for g, key in zip(guess, keys):
            coord.coords[key] = g
        self.computeS(coord, compareMode='CONTINUUM')
        return coord.S

    def veilingTest(self, guess, coord, junk):
        coord.coords["r"] = guess
        self.computeS(coord, compareMode='LINES')
        return coord.S
        
    def fitPhysical(self, guess, coord, junk):
        coord.coords["T"] = guess[0]
        coord.coords["G"] = guess[1]
        coord.coords["B"] = guess[2]
        coord.coords["r"] = guess[3]
        self.computeS(coord, compareMode="LINES")
        print "%10.3f %10.3f %10.3f %10.3f %10.3f" % (coord.coords["T"], coord.coords["G"], coord.coords["B"], coord.coords["r"],  coord.S)
        self.convergencePoints.append(copy.deepcopy(coord))
        return coord.S

    def findBestFitVeiling(self, coord):
        '''
            This procedure should find the best fit veiling and continuum values for the fidiucial value
        '''
        cont_guess = []
        for i in self.interpolated_model.keys():
            cont_guess.append(1.0)
        cont_guess = numpy.array(cont_guess)
        cont_value = scipy.optimize.fmin(self.continuumTest, cont_guess, (coord, self.interpolated_model.keys()),disp=False)
        #print cont_value
        veil_guess = numpy.array([0.0])
        veiling = scipy.optimize.fmin(self.veilingTest, veil_guess, (coord, 1), disp = False)
        #veiling = scipy.optimize.fmin_bfgs(self.veilingTest, veil_guess, args=(coord, 1), disp = False)
        #print veiling
        #print coord.coords["T"], coord.coords["G"], coord.coords["B"], coord.S
        return coord

    def findBestFitModel(self, coord):
        cont_guess = []
        for i in self.interpolated_model.keys():
            cont_guess.append(1.0)
        cont_guess = numpy.array(cont_guess)
        cont_value = scipy.optimize.fmin(self.continuumTest, cont_guess, (coord, self.interpolated_model.keys()),
        disp=False)
        print cont_value
        fit_target = coord.S/100.0
        params = [coord.coords["T"], coord.coords["G"], coord.coords["B"], coord.coords["r"]]
        physical_parameters = scipy.optimize.fmin(self.fitPhysical, params, (coord, 1), xtol=fit_target, ftol=fit_target)
        print 'Converged!'

    def allTest(self, guess, coord, junk):
        coord.coords["T"] = guess[0]
        coord.coords["G"] = guess[1]
        coord.coords["B"] = guess[2]
        coord.coords["r"] = guess[3]
        for g, key in zip(guess[4:], self.interpolated_model.keys()):
            coord.coords[key] = g
        #coord.checkLimits()
        self.computeS(coord, compareMode="LINES")
        print coord.dump()
        print coord.S
        return coord.S

    def newton_search(self, coord, plt=None):
        parameters = [coord.coords["T"], coord.coords["G"], coord.coords["B"], coord.coords["r"]]
        epsilon = numpy.array([25.0, 10.0, 0.1, 0.05])
        lb = numpy.array([3800, 400, 2.0, 0.0005])
        ub = numpy.array([4000, 500, 4.0, 0.5])
        for key in self.interpolated_model.keys():
            parameters.append(coord.coords[key])
            epsilon = numpy.append(epsilon, 0.002)
            lb = numpy.append(lb, 0.8)
            ub = numpy.append(ub, 1.2)
        best_fit = scipy.optimize.fmin(self.allTest, parameters, args=(coord, 1), full_output=True, ftol=coord.S/10.0,maxiter=50, xtol=coord.S/10.0)
        #best_fit = scipy.optimize.fmin_bfgs(self.allTest, parameters, args = (coord, 1), fprime=self.computeGradient,gtol=100.0)
        return coord

    def interpolatedModel(self, T, G, B):
        if ( (T < 2500) | (T > 6000) | (G < 300) | (G > 500) | (B < 0) | (B > 4.0) ):
            self.interpolated_model[self.currFeat]["flux"]=numpy.zeros(len(self.interpolated_model[self.currFeat]["wl"]))
            self.interpolated_model[self.currFeat]["T"] = T
            self.interpolated_model[self.currFeat]["G"] = G
            self.interpolated_model[self.currFeat]["B"] = B
            new_y = self.interpolated_model[self.currFeat]["flux"]
        elif ( (abs(self.interpolated_model[self.currFeat]["T"]- T) > 2.0) |
        (abs(self.interpolated_model[self.currFeat]["G"]- G) > 2.0) |
        (abs(self.interpolated_model[self.currFeat]["B"]-B)>0.05) ):
            
            self.interpolated_model[self.currFeat]["T"] = T
            self.interpolated_model[self.currFeat]["G"] = G
            self.interpolated_model[self.currFeat]["B"] = B
            #choose bracketing temperatures
            if not(T in self.ranges["T"]):
                Tlow = max(self.ranges["T"][scipy.where(self.ranges["T"] <= T)])
                Thigh = min(self.ranges["T"][scipy.where(self.ranges["T"] >= T)])
            else:
                Tlow = Thigh = T

            #choose bracketing surface gravities
            if not(G in self.ranges["G"]):
                Glow = max(self.ranges["G"][scipy.where(self.ranges["G"] <= G)])
                Ghigh = min(self.ranges["G"][scipy.where(self.ranges["G"] >= G)])
            else:
                Glow = Ghigh = G

            #choose bracketing B-fields
            if not(B in self.ranges["B"]):
                Blow = max(self.ranges["B"][scipy.where(self.ranges["B"] <= B)])
                Bhigh  = min(self.ranges["B"][scipy.where(self.ranges["B"] >= B)])
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

            self.interpolated_model[self.currFeat]["flux"] = new_y
        else:
            new_y = self.interpolated_model[self.currFeat]["flux"]
        return new_y

    def readMOOGModel(self, T, G, B, **kwargs):
        df = self.modelBaseDir+'B_'+str(B)+'kG/'+str(self.features[self.currFeat]["num"])+'_T'+str(int(T))+'G'+str(int(G))+'_R2000'
        x_sm, y_sm = SpectralTools.read_2col_spectrum(df)
        if "axis" in kwargs:
            if kwargs["axis"] == 'y':
                return y_sm
            elif kwargs["axis"] == 'x':
                return x_sm
        else:
            return x_sm, y_sm

    def computeS(self, coord, returnSpectra=False, computeVeiling=False, compareMode="LINES", **kwargs):
        if returnSpectra:
            retval = {}
        if computeVeiling:
            self.findBestFitVeiling(coord)
            # Find best fit contiuum and veiling.  Maybe assume continuum = 1.0?
        else:
            chiSq = 0.0
            for num in self.interpolated_model.keys():
                self.currFeat = num
                x_sm = self.interpolated_model[num]["wl"]
                new_wl = self.x_window[num] + self.features[num]["x_offset"]
                overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]
                if ( (abs(self.interpolated_model[self.currFeat]["T"]- coord.coords["T"]) > 2.0) |
                (abs(self.interpolated_model[self.currFeat]["G"]- coord.coords["G"]) > 2.0) |
                (abs(self.interpolated_model[self.currFeat]["B"]-coord.coords["B"])>0.05) ):
                    y_new = self.interpolatedModel(coord.coords["T"], coord.coords["G"], coord.coords["B"])
                    self.synthetic_spectrum[num] = self.binMOOGSpectrum(y_new, x_sm, new_wl[overlap])
                if ( (self.veiling_model[num]["r"] != coord.coords["r"]) | (self.veiling_model[num]["T"] != coord.coords["T"]) ):
                    self.veiling_model[num]["r"] = coord.coords["r"]
                    self.veiling_model[num]["T"] = coord.coords["T"]
                    self.veiling_model[num]["flux"] = self.compute_Excess(new_wl[overlap], coord.coords["T"], coord.coords["r"])
                veiled=(self.synthetic_spectrum[num]*coord.coords[num]+self.veiling_model[num]["flux"])/(self.veiling_model[num]["flux"]+1.0)

                if returnSpectra:    # We are simply returning the different spectra
                    retval[num] = {"wl":new_wl[overlap], "obs":self.flat[self.currFeat][overlap], "veiled":veiled, "noise":self.z[self.currFeat][overlap]}
                #'''
                if ( (num == 1) & ("plt" in kwargs)):
                    obs = Gnuplot.Data(new_wl[overlap], self.flat[num][overlap], with_='lines')
                    sim = Gnuplot.Data(new_wl[overlap], self.synthetic_spectrum[num], with_='lines')
                    veil = Gnuplot.Data(new_wl[overlap], veiled, with_='lines')
                    kwargs["plt"].plot(obs, sim, veil)
                #'''
                if (compareMode == 'LINES'):
                    chiSq += self.calcError(self.flat[num][overlap],veiled,self.z[num][overlap],new_wl[overlap],self.features[num]["comparePoints"])
                elif (compareMode == 'CONTINUUM'):
                    chiSq += self.calcError(self.flat[num][overlap],veiled,self.z[num][overlap],new_wl[overlap], self.features[num]["continuumPoints"])

            coord.addChiSquared(chiSq)
            if returnSpectra:
                return retval
    
    def compute_Excess(self, wl, Teff, veiling):
        excess_BB = self.veiling_SED(wl)
        star_BB = SpectralTools.blackBody(wl = wl/10000.0, T=Teff, outUnits="Energy")
        zp = SpectralTools.blackBody(wl=2.2/10000.0, T=Teff, outUnits="Energy")
        star_BB /= zp
        excess = excess_BB/star_BB*veiling
        
        return excess
    
    def computeDeriv(self, coord, key):
        coord_one = copy.deepcopy(coord)
        if key in ["T", "G", "B", "r"]:
            factor = 1.0
            while ((coord_one.coords[key] + self.delta[key]*factor) >= coord_one.limits[key][1]):
                factor -= 0.2
            coord_one.coords[key] += self.delta[key]*factor
            delta_one = self.delta[key]*factor
        else:
            coord_one.coords[key] += self.delta["dy"]
            delta_one = self.delta["dy"]

        self.computeS(coord_one)

        coord_two = copy.deepcopy(coord)
        if key in ["T", "G", "B", "r"]:
            factor = 1.0
            while ((coord_two.coords[key] - self.delta[key]*factor) <= coord_two.limits[key][0]):
                factor -= 0.2
            coord_two.coords[key] -= self.delta[key]*factor
            delta_two = self.delta[key]*factor
        else:
            coord_two.coords[key] -= self.delta["dy"]
            delta_two = self.delta["dy"]
            
        self.computeS(coord_two)

        delta = (delta_one + delta_two)/2.0
        return (coord_one.S-coord_two.S)/(2*delta)

    def compute2ndDeriv(self, c, keys, i, j):
        coords = copy.deepcopy(c)
        key_i = keys[i]
        key_j = keys[j]
        ifactor = 1.0
        if ( key_i in ["T", "G", "B"]):
            if ((coords.coords[key_i] - self.delta[key_i]*2*ifactor) <= coords.limits[key_i][0]):
                coords.coords[key_i] += self.delta[key_i]*2.5*ifactor
            if ((coords.coords[key_i]+self.delta[key_i]*2*ifactor) >= coords.limits[key_i][1]):
                coords.coords[key_i] -= self.delta[key_i]*2.5*ifactor
        coord_one = copy.deepcopy(coords)
        coord_two = copy.deepcopy(coords)
        coord_three = copy.deepcopy(coords)
        coord_four = copy.deepcopy(coords)
        if (key_i in self.interpolated_model.keys()):
            coord_one.coords[key_i] += self.delta["dy"]*ifactor
            coord_three.coords[key_i] += self.delta["dy"]*ifactor
            coord_two.coords[key_i] -= self.delta["dy"]*ifactor
            coord_four.coords[key_i] -= self.delta["dy"]*ifactor
            denominator = self.delta["dy"]*ifactor
        else:
            coord_one.coords[key_i] += self.delta[key_i]*ifactor
            coord_three.coords[key_i] += self.delta[key_i]*ifactor
            coord_two.coords[key_i] -= self.delta[key_i]*ifactor
            coord_four.coords[key_i] -= self.delta[key_i]*ifactor
            denominator = self.delta[key_i]*ifactor

        jfactor = 1.0
        if ( key_j in ["T", "G", "B"]):
            if ((coords.coords[key_j] - self.delta[key_j]*jfactor) <= coords.limits[key_j][0]):
                coords.coords[key_j] += self.delta[key_j]*jfactor
            if  ((coords.coords[key_j]+self.delta[key_j]*jfactor) >= coords.limits[key_j][1]):
                coords.coords[key_j] -= self.delta[key_j]*jfactor
        if (key_j in self.interpolated_model.keys()):
            coord_one.coords[key_j] += self.delta["dy"]*jfactor
            coord_two.coords[key_j] += self.delta["dy"]*jfactor
            coord_three.coords[key_j] -= self.delta["dy"]*jfactor
            coord_four.coords[key_j] -= self.delta["dy"]*jfactor
            denominator *= self.delta["dy"]*jfactor
        else:
            coord_one.coords[key_j] += self.delta[key_j]*jfactor
            coord_two.coords[key_j] += self.delta[key_j]*jfactor
            coord_three.coords[key_j] -= self.delta[key_j]*jfactor
            coord_four.coords[key_j] -= self.delta[key_j]*jfactor
            denominator *= self.delta[key_j]*jfactor

        coord_one.checkLimits()
        coord_two.checkLimits()
        coord_three.checkLimits()
        coord_four.checkLimits()
        
        self.computeS(coord_one)
        self.computeS(coord_two)
        self.computeS(coord_three)
        self.computeS(coord_four)

        return (coord_one.S - coord_two.S - coord_three.S + coord_four.S)/(4*denominator)

    def computeGradient(self, guess, coord, junk):
        print 'Calculating Gradient'
        coord.coords["T"] = guess[0]
        coord.coords["G"] = guess[1]
        coord.coords["B"] = guess[2]
        coord.coords["r"] = guess[3]
        for g, key in zip(guess[4:], self.interpolated_model.keys()):
            coord.coords[key] = g
        G = numpy.zeros(len(guess))

        for i in range(len(guess)):
            G[i] = self.computeDeriv(coord, coord.coords.keys()[i])

        print 'Gradient = ', G
        return G

    def computeHessian(self, coords):
        new_coords = copy.deepcopy(coords)
        '''
             i = 0, 1, 2 -> T, G, B
             i = 3 = veiling
             i = 4 ... n -> dy_i, dy_(n-4)
        '''
        keys = {}
        n = 0
        for i in ["T", "G", "B", "r"]:
            if self.floaters[i] == True:
                keys[n] = i
                n += 1
        if self.floaters["dy"] == True:
            for key in self.interpolated_model.keys():
                keys[n] = key
                n += 1
        H = numpy.zeros([n,n])

        for i in range(n):
            for j in numpy.arange(n - i) + i:
                H[i,j] = H[j,i] = self.compute2ndDeriv(new_coords, keys, i, j)
                print i, j, H[i,j]

        self.Hessian = numpy.matrix(H)

    def countPoints(self, minimum):
        points = 0.0
        for feat in self.interpolated_model.keys():
            wl = self.interpolated_model[feat]["wl"]
            comparePoints = self.features[feat]["comparePoints"]
            
            bm = []
            for region in comparePoints:
                bm.extend(scipy.where( (wl > region[0]) & (wl < region[1]) )[0])

            points += len(bm)

        return points


    def computeCovariance(self, minimum):
        self.computeHessian(minimum)
        self.computeS(minimum)
        self.covariance = 2*self.Hessian.getI()*(minimum.S/(self.countPoints(minimum)-minimum.n_dims))
        return self.covariance

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

    def loadObservations(self, wl, flux, error):
        self.x_window = {}
        self.flat = {}
        self.z = {}
        self.interpolated_model = {}
        self.veiling_model = {}
        self.synthetic_spectrum = {}
        contLevels = {}

        for num in self.features.keys():     #Sets up the parameters for the features covered by the spectra
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
                self.interpolated_model[num] = {"T":4000.0, "G":400.0, "B":0.0, "wl":x_sm, "flux":y_sm}
                self.veiling_model[num] = {"T":0.0, "r":0.0, "flux":[]}
                self.synthetic_spectrum[num] = []
                self.findWavelengthShift(x_window, flat, x_sm, y_sm)      #Finds the wavelength shift


    def computeErrors(self, wl, flux, error, best_point):
        self.x_window = {}
        self.flat = {}
        self.z = {}
        self.interpolated_model = {}
        self.veiling_model = {}
        self.synthetic_spectrum = {}
        contLevels = {}

        for num in self.features.keys():     #Sets up the parameters for the features covered by the spectra
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
                self.interpolated_model[num] = {"T":4000.0, "G":400.0, "B":0.0, "wl":x_sm, "flux":y_sm}
                self.veiling_model[num] = {"T":0.0, "r":0.0, "flux":[]}
                self.synthetic_spectrum[num] = []
                self.findWavelengthShift(x_window, flat, x_sm, y_sm)      #Finds the wavelength shift

                contLevels[num] = 1.0
        
        covariance = self.computeCovariance(best_point)

        return covariance

    def fitSpectrum(self, wl, flux, error, plt, **kwargs):
        outfile = kwargs["outfile"]
        # Gets the initial guess coordinates for both features
        guess_coords = {}
        self.x_window = {}
        self.flat = {}
        self.z = {}
        self.interpolated_model = {}
        self.veiling_model = {}
        self.synthetic_spectrum = {}
        contLevels = {}

        for num in self.features.keys():    #Sets up the parameters for the features covered by the spectra
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
                self.interpolated_model[num] = {"T":4000.0, "G":400.0, "B":0.0, "wl":x_sm, "flux":y_sm}
                self.veiling_model[num] = {"T":0.0, "r":0.0, "flux":[]}
                self.synthetic_spectrum[num] = []
                self.findWavelengthShift(x_window, flat, x_sm, y_sm)      #Finds the wavelength shift

                contLevels[num] = 1.0
                
        gridPoints = []
        chiSquaredMap = self.dataBaseDir+'gridSearch/'+kwargs["outfile"]+'.chisq'
        if not(os.path.exists(chiSquaredMap)):
            out=open(chiSquaredMap, 'w')
            for T in self.coarseRanges["T"]:
                for G in self.coarseRanges["G"]:
                    for B in self.coarseRanges["B"]:
                        gp = gridPoint({"T":T, "G":G, "B":B}, contLevels, 0.0)
                        self.computeS(gp, computeVeiling=True)
                        gridPoints.append(gp)
                        self.convergencePoints.append(gp)
                        #min_point = min(gridPoints)
                        #print min_point.dump()
                        #self.computeS(min_point, plt=plt)
                        print "Last point: T =" + str(T)+", G ="+str(G)+", B ="+str(B)+", S ="+str(gridPoints[-1].S)+", r_2.2 ="+str(gridPoints[-1].coords["r"])
                        out.write('%10.3f%10.3f%10.3f%10.3f%10.3e\n' % (T, G, B, gp.coords["r"], gp.S) )
            out.close()
        else:
            data = open(chiSquaredMap, 'r').readlines()
            for line in data:
                l = line.split()
                gp = gridPoint({"T":float(l[0]), "G":float(l[1]), "B":float(l[2])}, contLevels, float(l[3]))
                gp.S = float(l[4])
                gridPoints.append(gp)
                self.convergencePoints.append(gp)
        
        self.plotContour(gridPoints, outfile=outfile)
        fineRange = self.zoomIn(gridPoints)

        fineGrid = []
        chiSquaredZoomMap = self.dataBaseDir+'gridSearch/'+kwargs["outfile"]+'.chisq.zoom'
        if not(os.path.exists(chiSquaredZoomMap)):
            out=open(chiSquaredZoomMap, 'w')
            for T in fineRange["T"]:
                for G in fineRange["G"]:
                    for B in fineRange["B"]:
                        gp = gridPoint({"T":T, "G":G, "B":B}, contLevels, 0.0)
                        self.computeS(gp, computeVeiling=True)
                        fineGrid.append(gp)
                        self.convergencePoints.append(gp)
                        print "Last point: T =" + str(T)+", G ="+str(G)+", B ="+str(B)+", S ="+str(fineGrid[-1].S)+", r_2.2 ="+str(fineGrid[-1].coords["r"])
                        out.write('%10.3f%10.3f%10.3f%10.3f%10.3e\n' % (T, G, B, gp.coords["r"], gp.S) )
            out.close()
        else:
            data = open(chiSquaredZoomMap, 'r').readlines()
            for line in data:
                l = line.split()
                gp = gridPoint({"T":float(l[0]), "G":float(l[1]), "B":float(l[2])}, contLevels, float(l[3]))
                gp.S = float(l[4])
                fineGrid.append(gp)
                self.convergencePoints.append(gp)

        self.plotContour(fineGrid, mode="fine", outfile = outfile)
        order = numpy.argsort(fineGrid)
        T = []
        G = []
        B = []
        r = []
        for i in order[0:10]:
            T.append(fineGrid[order[i]].coords["T"])
            G.append(fineGrid[order[i]].coords["G"])
            B.append(fineGrid[order[i]].coords["B"])
            r.append(fineGrid[order[i]].coords["r"])
        initial_guess = gridPoint({"T":numpy.mean(T), "G":numpy.mean(G), "B":numpy.mean(B)}, contLevels, numpy.mean(r))
        #initial_guess = gridPoint({"T":4250, "G":420.0, "B":2.0}, contLevels, 0.8)
        print 'Preliminary Guess:'
        print initial_guess.dump()
        self.findBestFitModel(initial_guess)
        print 'Final Answer :'
        print initial_guess.dump()
        chiSquaredLogFile = open(self.dataBaseDir+'gridSearch/'+kwargs["outfile"]+'.chisq.log', 'w')
        pickle.dump(self.convergencePoints, chiSquaredLogFile)
        chiSquaredLogFile.close()
        self.saveFigures(initial_guess, outfile=outfile)
        #self.saveFigures(best_coords, outfile=outfile+'_B=0')
        #covariance = self.computeCovariance(initial_guess)

        return initial_guess#, covariance

    def zoomIn(self, gridPoints):
        order = numpy.argsort(gridPoints)
        T = []
        G = []
        B = []
        for i in order[0:10]:
            T.append(gridPoints[i].coords["T"])
            G.append(gridPoints[i].coords["G"])
            B.append(gridPoints[i].coords["B"])
        
        retval = {}
        retval["T"] = self.ranges["T"][scipy.where( (self.ranges["T"] >= min(T)) & (self.ranges["T"] <= max(T)) )[0]]
        retval["G"] = self.ranges["G"][scipy.where( (self.ranges["G"] >= min(G)) & (self.ranges["G"] <= max(G)) )[0]]
        retval["B"] = self.ranges["B"][scipy.where( (self.ranges["B"] >= min(B)) & (self.ranges["B"] <= max(B)) )[0]]
        return retval

    def plotContour(self, gps, mode="coarse", outfile='TEST'):
        T = []
        G = []
        B = []
        S = []
        for gp in gps:
            T.append(gp.coords["T"])
            G.append(gp.coords["G"])
            B.append(gp.coords["B"])
            S.append(gp.S)

        T = numpy.array(T)
        G = numpy.array(G)
        B = numpy.array(B)
        S = numpy.array(S)


        minval = min(S)
        maxval = max(S)
        T_coords = numpy.unique(T)
        G_coords = numpy.unique(G)

        for bfield in numpy.unique(B):
            bm = scipy.where(B == bfield)
            Z = S[bm].reshape(len(T_coords), len(G_coords))
            Tpts = T[bm].reshape(len(T_coords), len(G_coords))
            Gpts = G[bm].reshape(len(T_coords), len(G_coords))
            fig = pyplot.figure(0)
            pyplot.clf()
            pyplot.contour(Tpts, Gpts, Z, 25, extent=(min(T_coords), min(G_coords), max(T_coords), max(G_coords)))
            if mode=="fine":
                pyplot.savefig(outfile+"_B_"+str(bfield)+'kG_zoom.eps')
            else:
                pyplot.savefig(outfile+'_B_'+str(bfield)+'kG.eps')
            #print bfield
            #print Z
            #print Tpts
            #print Gpts

    def gridSearch(self, gp):
        gp.addChiSquared(self.computeS(gp, computeVeiling=True))

    def saveFigures(self, coords, **kwargs): 
        fig_width_pt = 246.0
        fig_width = 7.0  # inches
        inches_per_pt = 1.0/72.27
        pts_per_inch = 72.27
        golden_mean = (numpy.sqrt(5)-1.0)/2.0
        fig_height = fig_width*golden_mean
        fig_size = [fig_width, fig_height]
        params = {'backend' : 'ps',
          'axes.labelsize' : 12,
          'text.fontsize' : 12,
          'legend.fontsize' : 12,
          'xtick.labelsize' : 10,
          'ytick.labelsize' : 10,
          'text.usetex' : True,
          'figure.figsize' : fig_size}

        pyplot.rcParams.update(params)
        plotData = self.computeS(coords, returnSpectra = True)
        YSO_Name = kwargs["outfile"].replace('_', ' ')
        for num in plotData.keys():
            fig = pyplot.figure(0)
            pyplot.clf()
            plots = []
            plots.append(pyplot.plot(plotData[num]["wl"], plotData[num]["obs"], label='Observed', lw = 0.25, color='r'))
            plots.append(pyplot.plot(plotData[num]["wl"], plotData[num]["veiled"],label='Best Fit',lw = 0.25,color='b'))
            pyplot.xlabel(r'$\lambda$ ($\mu$m)')
            pyplot.ylabel(r'$F_{\lambda}$')
            pyplot.title(YSO_Name+' Feature :'+str(num))
            pyplot.legend( plots, ['Observed','Best Fit'], 'best')
            pyplot.savefig('plots/'+kwargs["outfile"]+'_f'+str(num)+'.eps')

