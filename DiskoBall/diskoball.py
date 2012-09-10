import numpy
import scipy
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FormatStrFormatter

class Angle:
    def __init__(self, line):
        l = line.split()
        self.n = int(l[0])
        self.az = [float(l[1]), float(l[2]), float(l[3])]
        self.longitude = [float(l[4]), float(l[5])]
        self.phi = float(l[6])
        self.chi = float(l[7])
        self.mu = float(l[8])

nrings = 23
ncells = 695
pi = numpy.pi
total_surface_area = 4.0*pi
cell_area = total_surface_area/ncells

dfI = '/home/deen/Data/MoogStokes/Verification_Data/testline/delo.spectrum_I'
dfContinuum = '/home/deen/Data/MoogStokes/Verification_Data/testline/delo.continuum'
dfAngles = '/home/deen/Data/MoogStokes/Verification_Data/testline/delo.angles'
dfMoog = '/home/deen/Data/MoogStokes/Verification_Data/testline/flux.moog'

Angles = open(dfAngles, 'r')
StokesI = open(dfI, 'r')
Continuum = open(dfContinuum, 'r')
Moog = open(dfMoog, 'r')

moog_wl = []
moog_fl = []

for line in Moog:
    l = line.split()
    if len(l) == 2:
        moog_wl.append(float(l[0]))
        moog_fl.append(float(l[1]))

ang_info = []

for line in Angles:
    ang_info.append(Angle(line))

wl = []
I = []
C = []

for line in StokesI:
    l = line.split()
    wl.append(float(l[0]))
    a = []
    for fluxes in l[1:]:
        a.append(float(fluxes))
    I.append(a)

wave = numpy.mean(wl)
if ((1.0/(wave/10000.0)) < 2.4):
   alpha = -0.023 + 0.292/(wave/10000.0)
else:
   alpha = -0.507 + 0.441/(wave/10000.0)

I = numpy.array(I)
I = I.transpose()

for line in Continuum:
    l = line.split()
    a = []
    for fluxes in l[1:]:
        a.append(float(fluxes))
    C.append(a)

C = numpy.array(C)
C = C.transpose()

fig = pyplot.figure(1)
fig.clear()
ax = fig.add_subplot(1,1,1)

r2d = 180.0/pi

final_spectrum = numpy.zeros(len(wl))
total_weight = 0.0

inclination = pi/2.0
T_I = numpy.matrix([[1.0, 0.0, 0.0],
        [0.0, numpy.cos(inclination), numpy.sin(inclination)],
        [0.0, -numpy.sin(inclination), numpy.cos(inclination)]])

emergent_vector = numpy.matrix([1.0, 0.0, 0.0])
mus = []
limb_darkenings = []
projected_areas = []
weights = []

for junk in zip(I, C, ang_info):
    azimuth = junk[2].az
    n_az_steps = int(azimuth[2]*r2d-azimuth[1]*r2d)
    azs = azimuth[1]+(numpy.arange(n_az_steps)+0.5)*(azimuth[2]-azimuth[1])/n_az_steps
    az1 = azimuth[1]+(numpy.arange(n_az_steps))*(azimuth[2]-azimuth[1])/n_az_steps
    az2 = azimuth[1]+(numpy.arange(n_az_steps)+1.0)*(azimuth[2]-azimuth[1])/n_az_steps
    longitude = junk[2].longitude
    dphi = longitude[1]
    n_phi_steps = int(dphi*r2d)
    phis = longitude[0]-dphi/2.0+(numpy.arange(n_phi_steps)+0.5)*dphi/n_phi_steps
    for az in zip(azs, az1, az2):
        T_rho = numpy.matrix([[0.0, 0.0, 1.0],
                [-numpy.cos(az[0]), numpy.sin(az[0]), 0.0],
                [numpy.sin(az[0]), numpy.cos(az[0]), 0.0]])
        daz = numpy.sin(az[2])-numpy.sin(az[1])
        area = daz*dphi
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

                mus.append(mu)
                projected_areas.append(projected_area*r2d*r2d)
                limb_darkenings.append(limb_darkening)
                weights.append(weight)

mus = numpy.arccos(numpy.array(mus))*r2d
projected_areas = numpy.array(projected_areas)
limb_darkenings = numpy.array(limb_darkenings)
weights = numpy.array(weights)

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

f1 = pyplot.figure(1)
f1.clear()
ax1 = f1.add_subplot(1,1,1)
ax1.scatter(mus, projected_areas, marker=',')
f1.savefig('ld.png')
