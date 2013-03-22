import SpectralTools
import SEDTools
import scipy
import numpy
import scipy.integrate
#import Gnuplot
import pickle
import time


def write_parfile(filename, **kwargs):

    df = open(filename, 'w')
    labels = {'terminal':'x11', 'strong':1, 'atmosphere':1, 'molecules':2,
            'lines':1, 'damping':1, 'freeform':0,
            'flux/int':0, 'plot':2, 'obspectrum':5}
    file_labels = {'summary_out':'summary.out', 'standard_out':'out1',
            'smoothed_out':'smoothed.out', 'lines_in':'linelist.input',
            'stronglines_in':'stronglines.input', 'model_in':'model.md',
            'observed_in':'observed.dat'}
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
    df.write('               '+str(kwargs["wl_start"])+' '
            +str(kwargs["wl_stop"])+' 0.02 1.00\n')
    df.write('plotpars       1\n')
    df.write('               '+str(kwargs["wl_start"])+' '
            +str(kwargs["wl_stop"])+' 0.02 1.00\n')
    df.write('               0.00 0.000 0.000 1.00\n')
    df.write('               g 0.150 0.00 0.00 0.00 0.00\n')
    
    df.close()

class periodicTable( object ):
    def __init__(self):
        self.Zsymbol_table = {}
        df = open('/home/deen/Code/python/AstroCode/MOOGConstants.dat', 'r')
        for line in df.readlines():
            l = line.split('-')
            self.Zsymbol_table[int(l[0])] = l[1].strip()
            self.Zsymbol_table[l[1].strip()] = int(l[0])
        df.close()

    def translate(self, ID):
        retval = self.Zsymbol_table[ID]
        return retval

class VALD_line( object ):
    def __init__(self, line1, line2, pt):
        l1 = line1.split(',')
        l2 = line2.split()
        self.element = pt.translate(l1[0].strip('\'').split()[0])
        self.ionization = int(l1[0].strip('\'').split()[1])-1
        self.species = self.element + self.ionization/10.0
        self.wl = float(l1[1])
        self.loggf = float(l1[2])
        self.expot_lo = float(l1[3])
        self.J_lo = float(l1[4])
        self.expot_hi = float(l1[5])
        self.J_hi = float(l1[6])
        self.g_lo = float(l1[7])
        self.g_hi = float(l1[8])
        self.g_eff = float(l1[9])
        self.raditive = float(l1[10])
        self.stark = float(l1[11])
        self.VdW = float(l1[12])
        self.DissE = -99.0
        self.transition = line2.strip().strip('\'')

        if (self.g_lo == 99.0):
            angmom = {"S":0, "P":1, "D":2, "F":3, "G":4, "H":5, "I":6, "K":7, "L":8, "M":9}
            n = 0
            for char in self.transition:
                if char.isdigit():
                    S = (float(char)-1.0)/2.0
                if ((char.isupper()) & (n < 2)):
                    n+=1
                    L = angmom[char]
                    if n == 1:
                        if (self.J_lo > 0.0):
                            self.g_lo = 1.5+(S*(S+1.0)-L*(L+1))/(2*self.J_lo*(self.J_lo+1))
                        else:
                            self.g_lo = 0.0
                    else:
                        if (self.J_hi > 0.0):
                            self.g_hi = 1.5+(S*(S+1.0)-L*(L+1))/(2*self.J_hi*(self.J_hi+1))
                        else:
                            self.g_hi = 0.0
            
        self.lower = Observed_Level(self.J_lo, self.g_lo, self.expot_lo)
        self.upper = Observed_Level(self.J_hi, self.g_hi, self.expot_lo+12400.0/self.wl)
        self.zeeman = {}
        self.zeeman["NOFIELD"] = [[self.wl], [self.loggf]]

    def zeeman_splitting(self, B, **kwargs):
        self.compute_zeeman_transitions(B, **kwargs)
        wl = []
        lgf = []
        for transition in self.pi_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight
                    *10.0**(self.loggf)))
        self.zeeman["pi"] = [numpy.array(wl), numpy.array(lgf)]

        wl = []
        lgf = []
        for transition in self.lcp_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight
                    *10.0**(self.loggf)))
        self.zeeman["lcp"] = [numpy.array(wl), numpy.array(lgf)]

        wl = []
        lgf = []
        for transition in self.rcp_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight
                    *10.0**(self.loggf)))
        self.zeeman["rcp"] = [numpy.array(wl), numpy.array(lgf)]

    def compute_zeeman_transitions(self, B, **kwargs):
        # Computes the splitting associated with the Zeeman effect
        
        bohr_magneton = 5.78838176e-5           # eV*T^-1
        hc = 12400                              # eV*Angstroems
        lower_energies = {}
        upper_energies = {}
        for mj in self.lower.mj:
            lower_energies[mj] = self.lower.E+mj*self.lower.g*bohr_magneton*B

        for mj in self.upper.mj:
            upper_energies[mj] = self.upper.E+mj*self.upper.g*bohr_magneton*B

        pi_transitions = []
        rcp_transitions = []
        lcp_transitions = []
        pi_weight = 0.0
        rcp_weight = 0.0
        lcp_weight = 0.0

        delta_J = self.upper.J - self.lower.J
        J1 = self.lower.J

        self.geff = (0.5*(self.lower.g+self.upper.g)  
              +0.25*(self.lower.g-self.upper.g)*(self.lower.J*(self.lower.J+1)-
              self.upper.J*(self.upper.J+1.0)))

        for mj in lower_energies.keys():
            if (delta_J == 0.0):
                if upper_energies.has_key(mj+1.0):  # delta_Mj = +1 sigma comp
                    weight = (J1-mj)*(J1+mj+1.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight += weight
                if upper_energies.has_key(mj):    # delta_Mj = 0 Pi component
                    weight= mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj, mj))
                    pi_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1+mj)*(J1-mj+1.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight += weight
            elif (delta_J == 1.0):
                if upper_energies.has_key(mj+1.0): # delta_Mj = +1 sigma comp
                    weight = (J1+mj+1.0)*(J1+mj+2.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight += weight
                if upper_energies.has_key(mj):  # delta_Mj = 0 Pi component
                    weight= (J1+1.0)**2.0 - mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj,mj))
                    pi_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1-mj+1.0)*(J1-mj+2.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight += weight
            elif (delta_J == -1.0):
                if upper_energies.has_key(mj+1.0): # delta_Mj = +1 sigma comp
                    weight = (J1-mj)*(J1-mj-1.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight += weight
                if upper_energies.has_key(mj):   # delta_Mj = 0 Pi component
                    weight= J1**2.0 - mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj, mj))
                    pi_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1+mj)*(J1+mj-1.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight += weight
                    
        for transition in rcp_transitions:
            transition.weight /= rcp_weight
        for transition in pi_transitions:
            transition.weight /= pi_weight
        for transition in lcp_transitions:
            transition.weight /= lcp_weight
            
        self.pi_transitions = pi_transitions
        self.lcp_transitions = lcp_transitions
        self.rcp_transitions = rcp_transitions

    def dump(self, **kwargs):
        if "out" in kwargs:
            out = kwargs["out"]
            """if self.DissE > 0:
                out.write('%10.3f%10.5f%10.3f%10.3f%20.3f\n' % (self.wl,
                    self.species,self.EP, self.loggf, self.DissE))
            elif kwargs["mode"].upper() == 'FULL':
                if ( (self.EP < 20.0) & (self.species % 1 <= 0.2) ):
                    out.write(
          '%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n' %
          (self.wl,self.species,self.EP,self.loggf, self.Jlow, self.Jup,
          self.glow,self.gup,self.geff,self.VdW))
            elif kwargs["mode"].upper() == 'MOOG':"""
            if kwargs["mode"].upper() == "MOOG":
                if ( (self.expot_lo < 20.0) & (self.species % 1 <= 0.2) ):
                    if ( self.DissE == -99.0 ):
                        for i in range(len(self.zeeman["pi"][0])):
                            out.write('%10.3f%10s%10.3f%10.3f' %
                                  (self.zeeman["pi"][0][i],
                                  self.species,self.expot_lo,self.zeeman["pi"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f\n'% (' ',0.0))
                            else:
                                out.write('%10.3f%20s%10.3f\n' %
                                        (self.VdW, ' ', 0.0))
                        for i in range(len(self.zeeman["lcp"][0])):
                            out.write('%10.3f%10s%10.3f%10.3f' %
                                (self.zeeman["lcp"][0][i],
                                self.species,self.expot_lo,self.zeeman["lcp"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f\n'% (' ',-1.0))
                            else:
                                out.write('%10.3f%20s%10.3f\n' %
                                        (self.VdW, ' ',-1.0))
                        for i in range(len(self.zeeman["rcp"][0])):
                            out.write('%10.3f%10s%10.3f%10.3f' %
                                (self.zeeman["rcp"][0][i],
                                self.species,self.expot_lo,self.zeeman["rcp"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f\n'% (' ',1.0))
                            else:
                                out.write('%10.3f%20s%10.3f\n' %
                                        (self.VdW, ' ', 1.0))
                    else:
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.expot_lo,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f\n' %
                                    (' ',self.DissE, 1.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f\n' %
                                    (self.VdW, self.DissE, 1.0))
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.expot_lo,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f\n' %
                                    (' ',self.DissE, 0.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f\n' %
                                    (self.VdW, self.DissE, 1.0))
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.expot_lo,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f\n' %
                                    (' ',self.DissE, -1.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f\n' %
                                    (self.VdW, self.DissE, 1.0))
        else:
            print self.species, self.wl, self.expot_lo, self.loggf, self

    def __lt__(self, other):
        if isinstance(other, float):
            return self.wl < other
        else:
            return self.wl < other.wl

    def __gt__(self, other):
        if isinstance(other, float):
            return self.wl > other
        else:
            return self.wl > other.wl

    def __le__(self, other):
        if isinstance(other, float):
            return self.wl <= other
        else:
            return self.wl <= other.wl

    def __ge__(self, other):
        if isinstance(other, float):
            return self.wl >= other
        else:
            return self.wl >= other.wl

    def __eq__(self, other):
        if isinstance(other, float):
            return self.wl == other
        else:
            distance = ((self.wl - other.wl)**2.0 +
                    (self.species - other.species)**2.0 +
                    (self.EP - other.EP)**2.0)**0.5
            if other.Jlow == -1:
                return ( distance < 0.01 )
            else:
                return ( (distance < 0.01) & (self.Jup == other.Jup)
                        & (self.Jlow == other.Jlow) )
                

class zeemanTransition( object ):
    def __init__(self, wavelength, weight, m_up, m_low):
        self.wavelength = wavelength
        self.weight = weight
        self.m_up = m_up
        self.m_low = m_low

    def __eq__(self, other):
        return ( (self.wavelength == other.wavelength) &
                (self.m_up == other.m_up) & (self.m_low == other.m_low) )

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
            self.Jup = -1.0
        if 'Jlow' in kwargs:
            self.Jlow = kwargs['Jlow']
        else:
            self.Jlow = -1.0
        if 'gup' in kwargs:
            self.gup = kwargs['gup']
        else:
            self.gup = 1.0
        if 'glow' in kwargs:
            self.glow = kwargs['glow']
        else:
            self.glow = 1.0
        if 'geff' in kwargs:
            self.geff = kwargs['geff']
        else:
            self.geff = 1.0
        if 'VdW' in kwargs:
            self.VdW = kwargs['VdW']
        else:
            self.VdW = 0.0
        if 'DissE' in kwargs:
            self.DissE = kwargs['DissE']
        else:
            self.DissE = -99.0

        self.lower = Observed_Level(self.Jlow, self.glow, self.EP)
        self.upper = Observed_Level(self.Jup, self.gup, self.EP+12400.0/self.wl)
        self.zeeman = {}
        self.linpol = {}
        self.lhcpol = {}
        self.rhcpol = {}
        self.zeeman["NOFIELD"] = [[self.wl], [self.loggf]]

    def zeeman_splitting(self, B, **kwargs):
        self.compute_zeeman_transitions(B, **kwargs)
        wl = []
        lgf = []
        for transition in self.pi_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight
                    *10.0**(self.loggf)))
        self.zeeman["pi"] = [numpy.array(wl), numpy.array(lgf)]

        wl = []
        lgf = []
        for transition in self.lcp_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight
                    *10.0**(self.loggf)))
        self.zeeman["lcp"] = [numpy.array(wl), numpy.array(lgf)]

        wl = []
        lgf = []
        for transition in self.rcp_transitions:
            if (transition.weight > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight
                    *10.0**(self.loggf)))
        self.zeeman["rcp"] = [numpy.array(wl), numpy.array(lgf)]

    def compute_zeeman_transitions(self, B, **kwargs):
        # Computes the splitting associated with the Zeeman effect
        
        bohr_magneton = 5.78838176e-5           # eV*T^-1
        hc = 12400                              # eV*Angstroems
        lower_energies = {}
        upper_energies = {}
        for mj in self.lower.mj:
            lower_energies[mj] = self.lower.E+mj*self.lower.g*bohr_magneton*B

        for mj in self.upper.mj:
            upper_energies[mj] = self.upper.E+mj*self.upper.g*bohr_magneton*B

        pi_transitions = []
        rcp_transitions = []
        lcp_transitions = []
        pi_weight = 0.0
        rcp_weight = 0.0
        lcp_weight = 0.0

        delta_J = self.upper.J - self.lower.J
        J1 = self.lower.J

        self.geff = (0.5*(self.lower.g+self.upper.g)  
              +0.25*(self.lower.g-self.upper.g)*(self.lower.J*(self.lower.J+1)-
              self.upper.J*(self.upper.J+1.0)))

        for mj in lower_energies.keys():
            if (delta_J == 0.0):
                if upper_energies.has_key(mj+1.0):  # delta_Mj = +1 sigma comp
                    weight = (J1-mj)*(J1+mj+1.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight += weight
                if upper_energies.has_key(mj):    # delta_Mj = 0 Pi component
                    weight= mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj, mj))
                    pi_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1+mj)*(J1-mj+1.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight += weight
            elif (delta_J == 1.0):
                if upper_energies.has_key(mj+1.0): # delta_Mj = +1 sigma comp
                    weight = (J1+mj+1.0)*(J1+mj+2.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight += weight
                if upper_energies.has_key(mj):  # delta_Mj = 0 Pi component
                    weight= (J1+1.0)**2.0 - mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj,mj))
                    pi_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1-mj+1.0)*(J1-mj+2.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight += weight
            elif (delta_J == -1.0):
                if upper_energies.has_key(mj+1.0): # delta_Mj = +1 sigma comp
                    weight = (J1-mj)*(J1-mj-1.0)
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight,
                        mj+1, mj))
                    rcp_weight += weight
                if upper_energies.has_key(mj):   # delta_Mj = 0 Pi component
                    weight= J1**2.0 - mj**2.0
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), weight,
                        mj, mj))
                    pi_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1+mj)*(J1+mj-1.0)
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight,
                        mj-1, mj))
                    lcp_weight += weight
                    
        for transition in rcp_transitions:
            transition.weight /= rcp_weight
        for transition in pi_transitions:
            transition.weight /= pi_weight
        for transition in lcp_transitions:
            transition.weight /= lcp_weight
            
        self.pi_transitions = pi_transitions
        self.lcp_transitions = lcp_transitions
        self.rcp_transitions = rcp_transitions

    def dump(self, **kwargs):
        if "out" in kwargs:
            out = kwargs["out"]
            """if self.DissE > 0:
                out.write('%10.3f%10.5f%10.3f%10.3f%20.3f\n' % (self.wl,
                    self.species,self.EP, self.loggf, self.DissE))
            elif kwargs["mode"].upper() == 'FULL':
                if ( (self.EP < 20.0) & (self.species % 1 <= 0.2) ):
                    out.write(
          '%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n' %
          (self.wl,self.species,self.EP,self.loggf, self.Jlow, self.Jup,
          self.glow,self.gup,self.geff,self.VdW))
            elif kwargs["mode"].upper() == 'MOOG':"""
            if kwargs["mode"].upper() == "MOOG":
                if ( (self.EP < 20.0) & (self.species % 1 <= 0.2) ):
                    if ( self.DissE == -99.0 ):
                        for i in range(len(self.zeeman["pi"][0])):
                            out.write('%10.3f%10s%10.3f%10.3f' %
                                  (self.zeeman["pi"][0][i],
                                  self.species,self.EP,self.zeeman["pi"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f\n'% (' ',0.0))
                            else:
                                out.write('%10.3f%20s%10.3f\n' %
                                        (self.VdW, ' ', 0.0))
                        for i in range(len(self.zeeman["lcp"][0])):
                            out.write('%10.3f%10s%10.3f%10.3f' %
                                (self.zeeman["lcp"][0][i],
                                self.species,self.EP,self.zeeman["lcp"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f\n'% (' ',-1.0))
                            else:
                                out.write('%10.3f%20s%10.3f\n' %
                                        (self.VdW, ' ',-1.0))
                        for i in range(len(self.zeeman["rcp"][0])):
                            out.write('%10.3f%10s%10.3f%10.3f' %
                                (self.zeeman["rcp"][0][i],
                                self.species,self.EP,self.zeeman["rcp"][1][i]))
                            if self.VdW == 0:
                                out.write('%20s%20.3f\n'% (' ',1.0))
                            else:
                                out.write('%10.3f%20s%10.3f\n' %
                                        (self.VdW, ' ', 1.0))
                    else:
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.EP,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f\n' %
                                    (' ',self.DissE, 1.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f\n' %
                                    (self.VdW, self.DissE, 1.0))
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.EP,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f\n' %
                                    (' ',self.DissE, 0.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f\n' %
                                    (self.VdW, self.DissE, 1.0))
                        out.write('%10.3f%10.5f%10.3f%10.3f' %
                                (self.wl, self.species, self.EP,self.loggf))
                        if self.VdW == 0.0:
                            out.write('%10s%10.3f%20.3f\n' %
                                    (' ',self.DissE, -1.0))
                        else:
                            out.write('%10.3f%10.3f%20.3f\n' %
                                    (self.VdW, self.DissE, 1.0))
        else:
            print self.species, self.wl, self.EP, self.loggf, self

    def __lt__(self, other):
        if isinstance(other, float):
            return self.wl < other
        else:
            return self.wl < other.wl

    def __gt__(self, other):
        if isinstance(other, float):
            return self.wl > other
        else:
            return self.wl > other.wl

    def __le__(self, other):
        if isinstance(other, float):
            return self.wl <= other
        else:
            return self.wl <= other.wl

    def __ge__(self, other):
        if isinstance(other, float):
            return self.wl >= other
        else:
            return self.wl >= other.wl

    def __eq__(self, other):
        if isinstance(other, float):
            return self.wl == other
        else:
            distance = ((self.wl - other.wl)**2.0 +
                    (self.species - other.species)**2.0 +
                    (self.EP - other.EP)**2.0)**0.5
            if other.Jlow == -1:
                return ( distance < 0.01 )
            else:
                return ( (distance < 0.01) & (self.Jup == other.Jup)
                        & (self.Jlow == other.Jlow) )
                


class Observed_Level( object ):
    def __init__(self, J, g, E):
        self.E = E               # Energy of the level (in eV)
        self.J = J               # Total Angular Momentum quantum number
        if g != 99:
            self.g = g           # Lande g factor for the level
        else:
            self.g = 1.0         # Assume that the g-factor is 1.0 if
                                 # we don't have a measured/calcuated value

        # Calculate the Mj sublevels
        self.mj = numpy.arange(self.J, (-1.0*self.J)-0.5, step = -1)

class Strong_Line( object ):
    def __init__(self, wl_o, zeeman, species, ep, loggf, VdW):
        self.wl = wl_o
        self.zeeman_components = zeeman
        self.species= species
        self.ep = ep
        self.loggf = loggf
        self.VdW = VdW

    def dump_zeeman(self, out, **kwargs):
        if "MODE" in kwargs:
            if kwargs["MODE"].upper() == 'TRANS':
                order = numpy.argsort(self.zeeman_components[1][0])
                for i in range(len(order)):
                    out.write('%10.3f%10s%10.3f%10.3f' % (self.zeeman_components[1][0][order[i]], self.species, self.ep,
                    self.zeeman_components[1][1][order[i]]))
                    if self.VdW == 0:
                        out.write('\n')
                    else:
                        out.write('%10.3f\n' % self.VdW)
            elif kwargs["MODE"].upper() == 'LONG':
                order = numpy.argsort(self.zeeman_components[0][0])
                for i in range(len(order)):
                    out.write('%10.3f%10s%10.3f%10.3f' % (self.zeeman_components[0][0][order[i]], self.species, self.ep,
                    self.zeeman_components[0][1][order[i]]))
                    if self.VdW == 0:
                        out.write('\n')
                    else:
                        out.write('%10.3f\n' % self.VdW)
            else:
                print 'Error!  Valid options are "TRANS" and "LONG"'
        else:
            print 'Error!  MODE variable is required!'


    def dump_single(self, out):
        out.write('%10.3f%10s%10.3f%10.3f' % (self.wl, self.species, self.ep, self.loggf))
        if self.VdW == 0.0:
            out.write('\n')
        else:
            out.write('%10.3f\n' % self.VdW)
