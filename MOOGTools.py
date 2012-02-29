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
        df = open('/home/deen/Code/python/StarFormation/AstroCode/MOOGConstants.dat', 'r')
        for line in df.readlines():
            l = line.split('-')
            self.Zsymbol_table[int(l[0])] = l[1].strip()
            self.Zsymbol_table[l[1].strip()] = int(l[0])
        df.close()

    def translate(self, ID):
        retval = self.Zsymbol_table[ID]
        return retval

class zeemanTransition( object ):
    def __init__(self, wavelength, weight_para, weight_perp, m_up, m_low):
        self.wavelength = wavelength
        self.weight_para = weight_para
        self.weight_perp = weight_perp
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
            if (transition.weight_perp > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight_perp
                    *10.0**(self.loggf)))
        self.zeeman["pi"] = [numpy.array(wl), numpy.array(lgf)]

        wl = []
        lgf = []
        for transition in self.lcp_transitions:
            if (transition.weight_para > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight_para
                    *10.0**(self.loggf)))
        self.zeeman["lcp"] = [numpy.array(wl), numpy.array(lgf)]

        wl = []
        lgf = []
        for transition in self.rcp_transitions:
            if (transition.weight_para > 0):
                wl.append(transition.wavelength)
                lgf.append(numpy.log10(transition.weight_para
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

        transitions = []
        pi_transitions = []
        rcp_transitions = []
        lcp_transitions = []
        total_weight = 0.0

        delta_J = self.upper.J - self.lower.J
        J1 = self.lower.J

        self.geff = 0.5* (self.lower.g+self.upper.g) + 
              0.25*(self.lower.g-self.upper.g)*(self.lower.J*(self.lower.J+1) -
              self.upper.J*(self.upper.J+1.0))

        for mj in lower_energies.keys():
            if (delta_J == 0.0):
                if upper_energies.has_key(mj+1.0):  # delta_Mj = +1 sigma comp
                    weight = (J1-mj)*(J1+mj+1.0)
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                        weight/4.0, mj+1, mj))
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                        0.0, mj+1, mj))
                    total_weight += weight/2.0
                if upper_energies.has_key(mj):    # delta_Mj = 0 Pi component
                    weight= mj**2.0
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), 0.0, weight,
                        mj, mj))
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), 0.0, weight,
                        mj, mj))
                    total_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1+mj)*(J1-mj+1.0)
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                        weight/4.0, mj-1, mj))
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                        0.0, mj-1, mj))
                    total_weight += weight/2.0
            elif (delta_J == 1.0):
                if upper_energies.has_key(mj+1.0): # delta_Mj = +1 sigma comp
                    weight = (J1+mj+1.0)*(J1+mj+2.0)
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                        weight/4.0, mj+1, mj))
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                        0.0, mj+1, mj))
                    total_weight += weight/2.0
                if upper_energies.has_key(mj):  # delta_Mj = 0 Pi component
                    weight= (J1+1.0)**2.0 - mj**2.0
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), 0.0, weight,
                        mj, mj))
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), 0.0, weight,
                        mj,mj))
                    total_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1-mj+1.0)*(J1-mj+2.0)
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                        weight/4.0, mj-1, mj))
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                        0.0, mj-1, mj))
                    total_weight += weight/2.0
            elif (delta_J == -1.0):
                if upper_energies.has_key(mj+1.0): # delta_Mj = +1 sigma comp
                    weight = (J1-mj)*(J1-mj-1.0)
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                        weight/4.0, mj+1, mj))
                    rcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                        0.0, mj+1, mj))
                    total_weight += weight/2.0
                if upper_energies.has_key(mj):   # delta_Mj = 0 Pi component
                    weight= J1**2.0 - mj**2.0
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), 0.0, weight,
                        mj, mj))
                    pi_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj]-lower_energies[mj]), 0.0, weight,
                        mj, mj))
                    total_weight += weight
                if upper_energies.has_key(mj-1.0): # delta_Mj = -1 sigma comp
                    weight = (J1+mj)*(J1+mj-1.0)
                    transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                        weight/4.0, mj-1, mj))
                    lcp_transitions.append(zeemanTransition(hc/
                        (upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                        0.0, mj-1, mj))
                    total_weight += weight/2.0
                    
        for transition in transitions:
            transition.weight_para /= (total_weight/2.0)
            transition.weight_perp /= (total_weight/2.0)

        for transition in rcp_transitions:
            transition.weight_para /= (total_weight/2.0)
        for transition in pi_transitions:
            transition.weight_perp /= (total_weight/2.0)
        for transition in lcp_transitions:
            transition.weight_para /= (total_weight/2.0)
            
        self.transitions = transitions
        self.pi_transitions = pi_transitions
        self.lcp_transitions = lcp_transitions
        self.rcp_transitions = rcp_transitions

    def dump(self, **kwargs):
        if "out" in kwargs:
            out = kwargs["out"]
            if self.DissE > 0:
                out.write('%10.3f%10.5f%10.3f%10.3f%20.3f\n' % (self.wl, self.species,self.EP, self.loggf, self.DissE))
            elif kwargs["mode"].upper() == 'FULL':
                if ( (self.EP < 20.0) & (self.species % 1 <= 0.2) ):
                    out.write('%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n' % (self.wl,self.species,self.EP,self.loggf, self.Jlow, self.Jup,self.glow,self.gup,self.geff,self.VdW))
            elif kwargs["mode"].upper() == 'MOOG':
                if ( (self.EP < 20.0) & (self.species % 1 <= 0.2) ):
                    if ( ("zeeman" in kwargs) & (self.DissE == -99.0) ):
                        if (kwargs["zeeman"] == 'straight'):
                            for i in range(len(self.zeeman["pi"][0])):
                                out.write('%10.3f%10s%10.3f%10.3f' %
                                  (self.zeeman["pi"][0][i],
                                  self.species,self.EP,self.zeeman["pi"][1][i]))
                                if self.VdW == 0:
                                    out.write('%20s%10.3f\n'% (' ',0.0))
                                else:
                                    out.write('%10.3f%10s%10.3f\n' % (self.VdW, ' ', 0.0))
                            for i in range(len(self.zeeman["lcp"][0])):
                                out.write('%10.3f%10s%10.3f%10.3f' % (self.zeeman["lcp"][0][i],
                                self.species,self.EP,self.zeeman["lcp"][1][i]))
                                if self.VdW == 0:
                                    out.write('%20s%10.3f\n'% (' ',-1.0))
                                else:
                                    out.write('%10.3f%10s%10.3f\n' % (self.VdW, ' ',-1.0))
                            for i in range(len(self.zeeman["rcp"][0])):
                                out.write('%10.3f%10s%10.3f%10.3f' % (self.zeeman["rcp"][0][i],
                                self.species,self.EP,self.zeeman["rcp"][1][i]))
                                if self.VdW == 0:
                                    out.write('%20s%10.3f\n'% (' ',1.0))
                                else:
                                    out.write('%10.3f%10s%10.3f\n' % (self.VdW, ' ', 1.0))
                        else:
                            order = numpy.argsort(self.zeeman[kwargs["zeeman"]][0])
                            for i in range(len(order)):
                                out.write('%10.3f%10s%10.3f%10.3f' % (self.zeeman[kwargs["zeeman"]][0][order[i]],
                                self.species,self.EP,self.zeeman[kwargs["zeeman"]][1][order[i]]))
                                if self.VdW == 0:
                                    out.write('\n')
                                else:
                                    out.write('%10.3f\n' % self.VdW)
                    elif ( ("linpol" in kwargs) & (self.DissE == -99.0) ):
                        order = numpy.argsort(self.linpol[kwargs["linpol"]][0])
                        for i in range(len(order)):
                            out.write('%10.3f%10s%10.3f%10.3f' % (self.linpol[kwargs["linpol"]][0][order[i]],
                            self.species,self.EP,self.linpol[kwargs["linpol"]][1][order[i]]))
                            if self.VdW == 0:
                                out.write('\n')
                            else:
                                out.write('%10.3f\n' % self.VdW)
                    elif ( ("lhcpol" in kwargs) & (self.DissE == -99.0) ):
                        order = numpy.argsort(self.lhcpol[kwargs["lhcpol"]][0])
                        for i in range(len(order)):
                            out.write('%10.3f%10s%10.3f%10.3f' % (self.lhcpol[kwargs["lhcpol"]][0][order[i]],
                            self.species,self.EP,self.lhcpol[kwargs["lhcpol"]][1][order[i]]))
                            if self.VdW == 0:
                                out.write('\n')
                            else:
                                out.write('%10.3f\n' % self.VdW)
                    elif ( ("rhcpol" in kwargs) & (self.DissE == -99.0) ):
                        order = numpy.argsort(self.rhcpol[kwargs["rhcpol"]][0])
                        for i in range(len(order)):
                            out.write('%10.3f%10s%10.3f%10.3f' % (self.rhcpol[kwargs["rhcpol"]][0][order[i]],
                            self.species,self.EP,self.rhcpol[kwargs["rhcpol"]][1][order[i]]))
                            if self.VdW == 0:
                                out.write('\n')
                            else:
                                out.write('%10.3f\n' % self.VdW)
                    else:
                        out.write('%10.3f%10.3f%10.3f%10.3f' % (self.wl, self.species, self.EP,self.loggf))
                        if self.VdW != 0.0:
                            out.write('%10.3f' % self.VdW)
                        out.write('\n')
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
            distance = ((self.wl - other.wl)**2.0 + (self.species - other.species)**2.0 + (self.EP - other.EP)**2.0)**0.5
            if other.Jlow == -1:
                return ( distance < 0.01 )
            else:
                return ( (distance < 0.01) & (self.Jup == other.Jup) & (self.Jlow == other.Jlow) )
                


class Observed_Level( object ):
    def __init__(self, J, g, E):
        self.E = E               # Energy of the level (in eV)
        self.J = J               # Total Angular Momentum quantum number
        if g != 99:
            self.g = g           # Lande g factor for the level
        else:
            self.g = 1.0         # Assume that the g-factor is 1.0 if we don't have a measured/calcuated value

        self.mj = numpy.arange(self.J, (-1.0*self.J)-0.5, step = -1) # Calculate the Mj sublevels

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
