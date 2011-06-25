import SpectralTools
import SEDTools
import scipy
import numpy
import scipy.integrate
import Gnuplot
import pickle
import time


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

class periodicTable( object ):
    def __init__(self):
        self.Zsymbol_table = {}
        df = open('/home/deen/Code/python/StarFormation/AstroCode/MOOGConstants.dat', 'r')
        for line in df.readlines():
            l = line.split('-')
            self.Zsymbol_table[int(l[0])] = l[1].strip()
            self.Zsymbol_table[l[1].strip()] = int(l[0])
        df.close()

    def translate(self, ID):
        retval = self.Zsymbol_table[ID]
        return retval

class spectral_Line( object ):
    def __init__(self,wl, species, EP, loggf, **kwargs):
        self.PT = periodicTable()
        self.wl = wl
        if isinstance(species, str):
            tmp = species.split()
            self.species = self.PT.translate(tmp[0])+(float(tmp[1])-1.0)/10.0
        else:
            self.species = species
        self.EP = EP
        self.loggf = loggf
        self.transition = ''
        if 'Jup' in kwargs:
            self.Jup = kwargs['Jup']
        else:
            self.Jup = 0.5
        if 'Jlow' in kwargs:
            self.Jlow = kwargs['Jlow']
        else:
            self.Jlow = 0.0
        if 'gup' in kwargs:
            self.gup = kwargs['gup']
        else:
            self.gup = 1.2
        if 'glow' in kwargs:
            self.glow = kwargs['glow']
        else:
            self.glow = 1.2
        if 'geff' in kwargs:
            self.geff = kwargs['geff']
        else:
            self.geff = 1.2
        if 'VdW' in kwargs:
            self.VdW = kwargs['VdW']
        else:
            self.VdW = 0.0


    def dump(self, **kwargs):
        if "out" in kwargs:
            out = kwargs["out"]
            if kwargs["mode"].upper() == 'FULL':
                out.write('%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n' %
                (self.wl,self.PT(self.species),self.EP,self.loggf, self.Jlow, self.Jup,self.glow,self.gup,self.geff,self.VdW))
            elif kwargs["mode"].upper() == 'MOOG':
                out.write('%10.3f%10.3f%10.3f%10.3f' % (self.wl, self.species, self.EP,self.loggf))
                if self.VdW != 0.0:
                    out.write('%10.3f' % self.VdW)
                out.write('\n')
        else:
            print self.species, self.wl, self.EP, self.loggf, self
