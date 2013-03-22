import numpy
import scipy
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FormatStrFormatter
import SpectralTools
import scipy.interpolate

class Angle( object ):
    def __init__(self, line):
        l = line.split()
        self.n = int(l[0])
        self.az = [float(l[1]), float(l[2]), float(l[3])]
        self.longitude = [float(l[4]), float(l[5])]
        self.phi = float(l[6])
        self.chi = float(l[7])
        self.mu = float(l[8])

class Diskoball( object ):
    def __init__(self, datadir, basename):
        self.datadir = datadir
        self.basename = basename
        self.dfI = self.datadir+self.basename+'.spectrum_I'
        self.dfQ = self.datadir+self.basename+'.spectrum_Q'
        self.dfU = self.datadir+self.basename+'.spectrum_U'
        self.dfV = self.datadir+self.basename+'.spectrum_V'
        self.dfCont = self.datadir+self.basename+'.continuum'
        self.dfAngles = self.datadir+self.basename+'.angles'

        Angles = open(self.dfAngles, 'r')
        StokesI = open(self.dfI, 'r')
        StokesQ = open(self.dfQ, 'r')
        StokesU = open(self.dfU, 'r')
        StokesV = open(self.dfV, 'r')
        Continuum = open(self.dfCont, 'r')

        linecounter = 0
        self.ang_info = []
        for line in Angles:
            if linecounter == 0:
                l = line.split()
                self.ncells = int(l[0])
                self.nrings = int(l[1])
                self.inclination = float(l[2])
                self.PA = float(l[3])
                self.cell_area = 4.0*3.1415926/self.ncells
                linecounter +=1
            else:
                self.ang_info.append(Angle(line))


        wl = []
        I = []
        Q = []
        U = []
        V = []
        C = []
        
        for line in StokesI:
            l = line.split()
            wl.append(float(l[0]))
            a = []
            for fluxes in l[1:]:
                try:
                    a.append(float(fluxes))
                except:
                    print "Warning! I crazy format :", fluxes
                    a.append(float(0.0))
            
            I.append(a)

        for line in StokesQ:
            l = line.split()
            a = []
            for fluxes in l[1:]:
                try:
                    a.append(float(fluxes))
                except:
                    print "Warning! Q crazy format :", fluxes
                    a.append(float(0.0))

            Q.append(a)

        for line in StokesU:
            l = line.split()
            a = []
            for fluxes in l[1:]:
                try:
                    a.append(float(fluxes))
                except:
                    print "Warning! U crazy format :", fluxes
                    a.append(float(0.0))

            U.append(a)

        for line in StokesV:
            l = line.split()
            a = []
            for fluxes in l[1:]:
                try:
                    a.append(float(fluxes))
                except:
                    print "Warning! V crazy format :", fluxes
                    a.append(float(0.0))

            V.append(a)

        for line in Continuum:
            l = line.split()
            a = []
            for fluxes in l[1:]:
                try:
                    a.append(float(fluxes))
                except:
                    print "Warning! C crazy format :", fluxes
                    a.append(float(0.0))

            C.append(a)
        
        self.wl = numpy.array(wl)
        I = numpy.array(I)
        Q = numpy.array(Q)
        U = numpy.array(U)
        V = numpy.array(V)
        C = numpy.array(C)
        self.I = I.transpose()
        self.Q = Q.transpose()
        self.U = U.transpose()
        self.V = V.transpose()
        self.C = C.transpose()

        wave = numpy.mean(self.wl)
        if ((1.0/(wave/10000.0)) < 2.4):
            self.alpha = -0.023 + 0.292/(wave/10000.0)
        else:
            self.alpha = -0.507 + 0.441/(wave/10000.0)


    def interpolate(self, stepsize):
        self.wave = numpy.arange(self.wl[0], self.wl[-1], step=stepsize)
        fI = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_I, s=0)
        fQ = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_Q, s=0)
        fU = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_U, s=0)
        fV = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_V, s=0)
        fC = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_C, s=0)
        self.flux_I = fI(self.wave)
        self.flux_Q = fQ(self.wave)
        self.flux_U = fU(self.wave)
        self.flux_V = fV(self.wave)
        self.flux_C = fC(self.wave)


    def disko(self):
        r2d = 180.0/numpy.pi
        final_I = numpy.zeros(len(self.wl))
        final_Q = numpy.zeros(len(self.wl))
        final_U = numpy.zeros(len(self.wl))
        final_V = numpy.zeros(len(self.wl))
        final_C = numpy.zeros(len(self.wl))
        
        total_weight = 0.0
        T_I = numpy.matrix([[1.0, 0.0, 0.0],
              [0.0, numpy.cos(self.inclination), numpy.sin(self.inclination)],
              [0.0, -numpy.sin(self.inclination), numpy.cos(self.inclination)]])

        emergent_vector = numpy.matrix([1.0, 0.0, 0.0])

        for tile in zip(self.I, self.Q, self.U, self.V, self.C, self.ang_info):
            azimuth = tile[5].az
            n_az_steps = int(azimuth[2]*r2d-azimuth[1]*r2d)
            azs = azimuth[1]+(numpy.arange(n_az_steps)+0.5)*(azimuth[2]-
                    azimuth[1])/n_az_steps
            az1 = azimuth[1]+(numpy.arange(n_az_steps))*(azimuth[2]-
                    azimuth[1])/n_az_steps
            az2 = azimuth[1]+(numpy.arange(n_az_steps)+1.0)*(azimuth[2]-
                    azimuth[1])/n_az_steps
            longitude = tile[5].longitude
            dphi = longitude[1]
            n_phi_steps = int(dphi*r2d)
            phis = longitude[0]-dphi/2.0+(numpy.arange(n_phi_steps)+
                    0.5)*dphi/n_phi_steps
            for az in zip(azs, az1, az2):
                T_rho = numpy.matrix([[0.0, 0.0, 1.0],
                        [-numpy.cos(az[0]), numpy.sin(az[0]), 0.0],
                        [numpy.sin(az[0]), numpy.cos(az[0]), 0.0]])
                daz = numpy.sin(az[2])-numpy.sin(az[1])
                area = daz*dphi/n_phi_steps
                for phi in phis:
                    T_eta = numpy.matrix([
                            [numpy.cos(phi), -numpy.sin(phi), 0.0],
                            [numpy.sin(phi), numpy.cos(phi), 0.0],
                            [0.0, 0.0, 1.0]])
                    surface_vector = T_I*T_eta*T_rho*emergent_vector.T
                    mu = surface_vector.A[2][0]
                    if (mu > 0.00001):
                        projected_area = area*mu#/(4.0*pi)
                        limb_darkening = (1.0-(1.0-mu**self.alpha))
                        weight = projected_area*limb_darkening
                        total_weight += weight
                        final_I = final_I + weight*tile[0]/tile[4]
                        final_Q = final_Q + weight*tile[1]/tile[4]
                        final_U = final_U + weight*tile[2]/tile[4]
                        final_V = final_V + weight*tile[3]/tile[4]
                        final_C = final_C + weight*tile[4]

        self.integrated_I = final_I/total_weight
        self.integrated_Q = final_Q/total_weight
        self.integrated_U = final_U/total_weight
        self.integrated_V = final_V/total_weight
        self.integrated_C = final_C/total_weight

    def save(self, outfile):
        SpectralTools.write_2col_spectrum(outfile+'.I', self.wave, self.flux_I)
        SpectralTools.write_2col_spectrum(outfile+'.Q', self.wave, self.flux_Q)
        SpectralTools.write_2col_spectrum(outfile+'.U', self.wave, self.flux_U)
        SpectralTools.write_2col_spectrum(outfile+'.V', self.wave, self.flux_V)
        SpectralTools.write_2col_spectrum(outfile+'.C', self.wave, self.flux_C)


