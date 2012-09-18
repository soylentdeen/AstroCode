import numpy
import scipy
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FormatStrFormatter
import SpectralTools
import scipy.interpolate

class Angle:
    def __init__(self, line):
        l = line.split()
        self.n = int(l[0])
        self.az = [float(l[1]), float(l[2]), float(l[3])]
        self.longitude = [float(l[4]), float(l[5])]
        self.phi = float(l[6])
        self.chi = float(l[7])
        self.mu = float(l[8])

class Diskoball:
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

        linecouner = 0
        self.ang_info = []
        for line in Angles:
            if linecounter == 0:
                l = line.split()
                self.ncells = int(l[0])
                self.nrings = int(l[1])
                self.inclination = float(l[2])
                self.PA = float(l[3])
                self.cell_area = 4.0*3.1415926/ncells
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
                a.append(float(fluxes))
            I.append(a)

        for line in StokesQ:
            l = line.split()
            a = []
            for fluxes in l[1:]:
                a.append(float(fluxes))
            Q.append(a)

        for line in StokesU:
            l = line.split()
            a = []
            for fluxes in l[1:]:
                a.append(float(fluxes))
            U.append(a)

        for line in StokesV:
            l = line.split()
            a = []
            for fluxes in l[1:]:
                a.append(float(fluxes))
            V.append(a)

        for line in Continuum:
            l = line.split()
            a = []
            for fluxes in l[1:]:
                a.append(float(fluxes))
            C.append(a)
        
        self.wl = numpy.array(wl)
        I = numpy.array(I)
        Q = numpy.array(Q)
        U = numpy.array(U)
        V = numpy.array(V)
        C = numpy.array(V)
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
        wave = numpy.arange(wl[0], wl[-1], step=0.001)
        fI = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_I, s=0)
        fQ = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_Q, s=0)
        fU = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_U, s=0)
        fV = scipy.interpolate.UnivariateSpline(self.wl, self.integrated_V, s=0)
        self.flux_I = fI(wave)
        self.flux_Q = fQ(wave)
        self.flux_U = fU(wave)
        self.flux_V = fV(wave)


    def disko(self):
        final_I = numpy.zeros(len(self.wl))
        final_Q = numpy.zeros(len(self.wl))
        final_U = numpy.zeros(len(self.wl))
        final_V = numpy.zeros(len(self.wl))
        
        total_weight = 0.0
        T_I = numpy.matrix([[1.0, 0.0, 0.0],
              [0.0, numpy.cos(inclination), numpy.sin(inclination)],
              [0.0, -numpy.sin(inclination), numpy.cos(inclination)]])

        emergent_vector = numpy.matrix([1.0, 0.0, 0.0])

SpectralTools.write_2col_spectrum('T50G5.delo.80.dat', wave, flux_stokes)
SpectralTools.write_2col_spectrum('T50G5.moog.80.dat', wave, flux_scalar)
        wave = 

fig = pyplot.figure(1)
fig.clear()
ax = fig.add_subplot(1,1,1)

r2d = 180.0/pi

final_spectrum = numpy.zeros(len(wl))
total_weight = 0.0

inclination = pi/2.0

for junk in zip(I, C, ang_info):
    azimuth = junk[2].az
    n_az_steps = int(azimuth[2]*r2d-azimuth[1]*r2d)
    azs = azimuth[1]+(numpy.arange(n_az_steps)+0.5)*(azimuth[2]-azimuth[1])/n_az_steps
    az1 = azimuth[1]+(numpy.arange(n_az_steps))*(azimuth[2]-azimuth[1])/n_az_steps
    az2 = azimuth[1]+(numpy.arange(n_az_steps)+1.0)*(azimuth[2]-azimuth[1])/n_az_steps
    longitude = junk[2].longitude
    dphi = longitude[1]
    n_phi_steps = int(dphi*r2d)
    phis=longitude[0]-dphi/2.0+(numpy.arange(n_phi_steps)+0.5)*dphi/n_phi_steps
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
                limb_darkening = (1.0-(1.0-mu**alpha))
                weight = projected_area*limb_darkening
                total_weight += weight
                final_spectrum = final_spectrum + weight*junk[0]/junk[1]


final_spectrum /= total_weight

majorFormatter = FormatStrFormatter('%10.1f')

ax.plot(wl, final_spectrum, label='MoogStokes', color = 'r')
ax.plot(moog_wl, numpy.array(moog_fl), label = 'MoogScalar', color='b')
ax.xaxis.set_major_formatter(majorFormatter)
ax.set_xbound(11990.0, 11993)
ax.set_ybound(1.0, 0.4)
ax.set_xlabel(r'Wavelength $\left(\AA\right)$')
ax.set_ylabel(r'$\frac{I}{C}$')
ax.set_title(r'DELO vs. Contribution Function Comparison : Full Disk')
ax.legend(loc=3)
fig.savefig('Diskint.png')

