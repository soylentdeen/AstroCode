import numpy
import scipy
import matplotlib.pyplot as pyplot

nrings = 23
ncells = 695
pi = numpy.pi
total_surface_area = 4.0*pi
cell_area = total_surface_area/ncells

for i in range(nrings):
    chi_start = i*pi/nrings
    chi_stop = (i+1)*pi/nrings
    azimuth = (chi_start+chi_stop)/2.0
    dchi = -numpy.cos(chi_stop) + numpy.cos(chi_start)
    ring_area = 2.0*pi*dchi
    cells_in_ring = int(ring_area/cell_area)
    cell_area_in_ring = ring_area/float(cells_in_ring)
    dphi = 2.0*pi/cells_in_ring
    print "azimuth = ", azimuth, "dchi = ", dchi, " #cells = ", cells_in_ring
    for j in range (cells_in_ring):
        longitude = -pi+(j+0.5)*dphi
        print "Longitude = ", longitude
    raw_input()
        
