import numpy
import scipy
import matplotlib.pyplot as pyplot

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

Angles = open(dfAngles, 'r')
StokesI = open(dfI, 'r')
Continuum = open(dfContinuum, 'r')

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
ax = fig.add_subplot(1,1,1)

r2d = 180.0/3.1415926

for junk in zip(I, C, ang_info):
    ax.clear()
    ax.plot(wl, junk[0]/junk[1])
    ax.set_ybound(1.0, 0.4)
    fig.show()
    mu = junk[2].mu
    az = junk[2].az
    longitude = junk[2].longitude
    daz = -numpy.cos(az[2]) + numpy.cos(az[1])
    print az[0]*r2d, longitude[0]*r2d, mu
    raw_input()
