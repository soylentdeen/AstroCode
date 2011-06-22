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

class gridPoint( object ):    # Object which contains information about a Chi-Squared grid point
    def __init__(self, TGBcoords, bandCoords):
        
        self.TGB = {"T":TGBcoords["T"], "G":TGBcoords["G"], "B":TGBcoords["B"]}
        self.bandCoords= {}
        self.n_dims = 3
        for i in len(bandCoords):
            self.bandCoords[i] = bandCoords[i]
            self.n_dims += len(bandCoords[i].keys())

class spectralSynthesizer( object ):   # low resolution
    def __init__(self):
        feat_num = [12, 57, 8]
        xstart = [1.190, 2.170, 2.260]
        xstop = [1.2100, 2.2112, 2.268]
        slope_start = [1.15, 2.100, 2.23]
        slope_stop = [1.25, 2.25, 2.29]
        strongLines = [[2.166, 2.2066], [2.166, 2.2066], [2.263, 2.267]]
        lineWidths = [[0.005, 0.005],[0.005, 0.005], [0.005, 0.005]]
        #comparePoints = [[[1.1816,1.1838],[1.1871,1.1900],[1.1942,1.1995],[1.2073,1.2087]], [[2.1765, 2.18], [2.1863,2.1906], [2.199, 2.2015], [2.2037, 2.210]], [[2.2525,2.2551],[2.2592, 2.2669], [2.2796, 2.2818]]]
        comparePoints = [[[1.1816,1.1995],[1.2073,1.2087]], [[2.1765, 2.18], [2.1863,2.1906], [2.199, 2.2015], [2.2037, 2.210]], [[2.2525,2.2551],[2.2592, 2.2669], [2.2796, 2.2818]]]

        self.modelBaseDir='/home/grad58/deen/Data/StarFormation/MOOG/zeeman/smoothed/'
        self.dataBaseDir='/home/grad58/deen/Data/StarFormation/TWA/bfields/'
        self.delta = {"T":200.0, "G":50.0, "B":0.5, "dy":0.03, "r":0.10]}    # [dT, dG, dB, d_dy, dr]
        self.delta_factor = 1.0
        self.limits = {"T":[2500.0, 6000.0], "G":[300.0, 500.0], "B":[0.0,4.0], "dy":[0.98, 1.02], "r":[0.0, 10.0]}
        self.floaters = numpy.array{"T":True, "G":True, "B":True, "dy":True, "r":True}
        self.features = []


        #for i in range(len(xstart)):
        for i in [0,1]:
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
        self.ranges = {"T":self.temps, "G":self.gravs, "B":self.bfields}

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
            
        self.features[self.currFeat]["x_offset"] = offset_computed

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
        df = self.modelBaseDir+'B_'+str(B)+'kG/feat_'+str(self.features[self.currFeat]["num"])+'_MARCS_T'+str(int(T))+'G'+str(int(G))+'_R2000'
        x_sm, y_sm = SpectralTools.read_2col_spectrum(df)
        if "axis" in kwargs:
            if kwargs["axis"] == 'y':
                return y_sm
            elif kwargs["axis"] == 'x':
                return x_sm
        else:
            return x_sm, y_sm

    def computeS(self, coordinates, **kwargs):
        TGBCoords = coordinates.TGBcoords
        #TGBCoords["T"] = T
        #TGBCoords["G"] = G
        #TGBCoords["B"] = B
        
        bandCoords = coordinates.bandCoords
        #bandCoords[*]["dy"] = dy
        #bandCoords[*]["r"] = r

        retval = 0.0
        for num in range(len(bandCoords))
            self.currFeat = num
            x_sm = self.features[self.currFeat]["wl"]
            y_new = self.interpolatedModel(TGBcoords["T"], TGBcoords["G"], TGBcoords["B"])

        
            new_wl = self.[self.currFeat]x_window + self.features[self.currFeat]["x_offset"]
            overlap = scipy.where( (new_wl > min(x_sm)) & (new_wl < max(x_sm)) )[0]

            synthetic_spectrum = self.binMOOGSpectrum(y_new, x_sm, new_wl[overlap])*coords["dy"]
            if "plot" in kwargs:
                obs = Gnuplot.Data(new_wl[overlap], self.flat[overlap], with_='lines')
                sim = Gnuplot.Data(new_wl[overlap], synthetic_spectrum, with_='lines')
                veil = Gnuplot.Data(new_wl[overlap], (synthetic_spectrum+coords["r"])/(coords["r"]+1.0), with_='lines')
                kwargs["plot"].plot(obs, sim, veil)
            retval += self.calcError(self.flat[self.currFeat][overlap],(synthetic_spectrum+bandCoords[self.currFeat]["r"])/(bandCoords[self.currFeat]["r"]+1.0),self.z[self.currFeat][overlap], new_wl[overlap], self.features[self.currFeat]["comparePoints"])

        return retval
    
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
                            self.z[overlap], new_wl[overlap], self.features[self.currFeat]["comparePoints"]))
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
        return numpy.array(initial_guess)


    def simplex(self, init_guess, plt):
        #simplex_coords = [init_guess]
        #simplex_values = [self.computeS(init_guess)]

        coords = []
        Svalues = []
        for key in self.floaters.keys():
            if self.floaters[key] == True:
                new_TGBcoords = init_guess.copy()
                bandCoords = {}
                if key in init_guess:
                    if (new_TGBcoords[key] + self.delta[key]) < self.limits[key][1]:
                        new_TGBcoords[key] += self.delta[key]
                    else:
                        new_TGBcoords[key] -= self.delta[key]
                    for num in range(len(self.features)):
                        bandCoords{num} = {"dy":1.0, "r":0.05}
                    simplexPoint = gridPoint(new_TGBcoords, bandCoords)
                    coords.append(simplexPoint)
                    Svalues.append(computeS(simplexPoint))
                else:
                    for num in range(len(self.features)):
                        bandCoords[num] = {"dy":1.0, "r":0.05}
                        if (bandCoords[key] + self.delta[key]) < self.limits[key][1]:
                            bandCoords[key] += self.delta[key]
                        else:
                            bandCoords[key] -= self.delta[key]
                        simplexPoint = gridPoint(new_TGBcoords, bandCoords)
                        coords.append(simplexPoint)
                        Svalues.append(computeS(simplexPoint))

        n_contractions = 0
        centroid = numpy.zeros(coords[0].n_dims)

        while (numpy.std(Svalues) > 0.5):
            order = numpy.argsort(Svalues)
            centroid = self.findCentroid(coords)
            print centroid

            # Reflect about centroid
            distance = centroid - best_coords[order[-1]]
            trial_1 = self.check_bounds(centroid + distance)
            S1 = self.computeS(trial_1)
            
            if (S1 > best_values[order[-1]]):    # Try longer path along distance
                trial_2 = self.check_bounds(centroid + 2.0*distance)
                S2 = self.computeS(trial_2)
                if (S2 > best_values[order[-1]]):    # Try half the distance
                    trial_3 = self.check_bounds(centroid + 0.5*distance)
                    S3 = self.computeS(trial_3)
                    if (S3 > best_values[order[-1]]):     # Shrink?
                        #self.delta_factor *= 2.0
                        if n_contractions <= 2:
                            for i in range(len(best_coords)):
                                distance = centroid-best_coords[order[i]]
                                best_coords[i] = self.check_bounds(best_coords[order[i]] - distance/2.0)
                                best_values[i] = self.computeS(best_coords[i])
                            print 'Contracted!'
                            n_contractions += 1
                        else:
                            break
                    else:
                        for i in range(len(order)-1):
                            best_coords[order[i]] = best_coords[order[i]]
                            best_values[order[i]] = best_values[order[i]]
                        best_coords[order[-1]] = trial_3
                        best_values[order[-1]] = self.computeS(trial_3)
                        print 'Trial 3'
                else:
                    for i in range(len(order)-1):
                        best_coords[order[i]] = best_coords[order[i]]
                        best_values[order[i]] = best_values[order[i]]
                    best_coords[order[-1]] = trial_2
                    best_coords[order[-1]] = self.computeS(trial_2)
                    print 'Trial 2'
            else:
                for i in range(len(order) -1):
                    best_coords[order[i]] = best_coords[order[i]]
                    best_values[order[i]] = best_values[order[i]]
                best_coords[order[-1]] = trial_1
                best_values[order[-1]] = self.computeS(trial_1)
                print 'Trial 1'
        
            print numpy.mean(best_values), numpy.std(best_values)

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
        return centroid
                        

    def fitSpectrum(self, wl, flux, error, plt1, plt2, **kwargs):
        retval = []
        outfile = kwargs["outfile"]
        # Gets the initial guess coordinates for both features
        guess_coords = []
        for feat, num in zip(self.features, range(len(self.features))):
            self.currFeat = num
            if (min(wl) < feat["xstart"]):
                x_window, flat, z = SEDTools.removeContinuum(wl, flux, error, feat["slope_star"], feat["slope_stop"], strongLines=feat["strongLines"], lineWidths=feat["linesWidhts"], errors=True)

                self.x_window[num] = x_window
                self.flat[num] = flat
                self.z[num] = z
                
                kwargs["outfile"]=self.dataBaseDir+outfile+'_feat_'+str(self.features[self.currFeat]["num"])+'.dat'
                guess_coords.append(self.gridSearch(MODE='FINAL'))
                
        initial_guess = guess_coords[0]
        best_coords = self.simplex(initial_guess, plt)
        retval.append(best_coords)
        print 'Best Fit Coordinates :', best_coords



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

                if kwargs["mode"] == 'FINAL':
                    x_sm, y_sm = self.readMOOGModel(4000.0, 400.0, 1.0)
                    self.features[self.currFeat]["wl"] = x_sm/10000.0
                    kwargs["outfile"]=self.dataBaseDir+outfile+'_feat_'+str(self.features[self.currFeat]["num"])+'.dat'
                    guess_coords=self.gridSearch(**kwargs)
                    if self.currFeat == 1:
                        guess_coords[0] = retval[-1][0]
                        self.floaters[0] = False
                    else:
                        self.floaters[0] = True
                    best_coords = self.simplex(guess_coords, plt)
                    retval.append(best_coords)
                    #best_coords = self.marquardt(chisq)
                    print 'Best Fit Coordinates :', best_coords
                else:
                    retval.append([0])
        return retval

