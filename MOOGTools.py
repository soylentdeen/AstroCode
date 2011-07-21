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

class zeemanTransition( object ):
    def __init__(self, wavelength, weight_para, weight_perp, m_up, m_low):
        self.wavelength = wavelength
        self.weight_para = weight_para
        self.weight_perp = weight_perp
        self.m_up = m_up
        self.m_low = m_low

    def __eq__(self, other):
        return ( (self.wavelength == other.wavelength) & (self.m_up == other.m_up) & (self.m_low == other.m_low) )

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
        self.zeeman["NOFIELD"] = [[self.wl], [self.loggf]]

    def zeeman_splitting(self, B, **kwargs):
        self.compute_zeeman_transitions(B, **kwargs)

    def compute_zeeman_transitions(self, B, **kwargs):    # Computes the splitting associated with the Zeeman effect
        bohr_magneton = 5.78838176e-5           # eV*T^-1
        hc = 12400                              # eV*Angstroems
        lower_energies = {}
        upper_energies = {}
        for mj in self.lower.mj:
            lower_energies[mj] = self.lower.E+mj*self.lower.g*bohr_magneton*B

        for mj in self.upper.mj:
            upper_energies[mj] = self.upper.E+mj*self.upper.g*bohr_magneton*B

        transitions = []
        sigma_transitions_para = []             # Energy of Sigma components in the parallel direction
        sigma_weights_para = []                 # Weights of sigma components in parallel direction
        sigma_transitions_perp = []             # Energy of Sigma component in perpendicular direction
        sigma_weights_perp = []                 # Weights of sigma components in perpendicular direction
        pi_transitions = []                     # Pi components are only seen in perpendicular direction
        pi_weights = []

        delta_J = self.upper.J - self.lower.J
        J1 = self.lower.J

        for mj in lower_energies.keys():
            if (delta_J == 0.0):
                if upper_energies.has_key(mj+1.0):    # delta_Mj = +1 sigma component
                    weight = (J1-mj)*(J1+mj+1.0)
                    transitions.append(zeemanTransition(hc/(upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                    weight/4.0, mj+1, mj))
                    sigma_transitions_perp.append(upper_energies[mj+1] - lower_energies[mj])
                    sigma_transitions_para.append(upper_energies[mj+1] - lower_energies[mj])
                    sigma_weights_perp.append((J1-mj)*(J1+mj+1.0)/4.0)
                    sigma_weights_para.append((J1-mj)*(J1+mj+1.0)/2.0)
                if upper_energies.has_key(mj):        # delta_Mj = 0 Pi component
                    weight= mj**2.0
                    transitions.append(zeemanTransition(hc/(upper_energies[mj]-lower_energies[mj]), 0.0, weight, mj,
                    mj))
                    pi_transitions.append(upper_energies[mj] - lower_energies[mj])
                    pi_weights.append(mj**2.0)
                if upper_energies.has_key(mj-1.0):    # delta_Mj = -1 sigma component
                    weight = (J1+mj)*(J1-mj+1.0)
                    transitions.append(zeemanTransition(hc/(upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                    weight/4.0, mj-1, mj))
                    sigma_transitions_perp.append(upper_energies[mj-1] - lower_energies[mj])
                    sigma_transitions_para.append(upper_energies[mj-1] - lower_energies[mj])
                    sigma_weights_perp.append((J1+mj)*(J1-mj+1.0)/4.0)
                    sigma_weights_para.append((J1+mj)*(J1-mj+1.0)/2.0)
            elif (delta_J == 1.0):
                if upper_energies.has_key(mj+1.0):    # delta_Mj = +1 sigma component
                    weight = (J1+mj+1.0)*(J1+mj+2.0)
                    transitions.append(zeemanTransition(hc/(upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                    weight/4.0, mj+1, mj))
                    sigma_transitions_perp.append(upper_energies[mj+1] - lower_energies[mj])
                    sigma_transitions_para.append(upper_energies[mj+1] - lower_energies[mj])
                    sigma_weights_perp.append((J1+mj+1.0)*(J1+mj+2.0)/4.0)
                    sigma_weights_para.append((J1+mj+1.0)*(J1+mj+2.0)/2.0)
                if upper_energies.has_key(mj):        # delta_Mj = 0 Pi component
                    weight= (J1+1.0)**2.0 - mj**2.0
                    transitions.append(zeemanTransition(hc/(upper_energies[mj]-lower_energies[mj]), 0.0, weight, mj,
                    mj))
                    pi_transitions.append(upper_energies[mj] - lower_energies[mj])
                    pi_weights.append((J1+1.0)**2.0 - mj**2.0)
                if upper_energies.has_key(mj-1.0):    # delta_Mj = -1 sigma component
                    weight = (J1-mj+1.0)*(J1-mj+2.0)
                    transitions.append(zeemanTransition(hc/(upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                    weight/4.0, mj-1, mj))
                    sigma_transitions_perp.append(upper_energies[mj-1] - lower_energies[mj])
                    sigma_transitions_para.append(upper_energies[mj-1] - lower_energies[mj])
                    sigma_weights_perp.append((J1-mj+1.0)*(J1-mj+2.0)/4.0)
                    sigma_weights_para.append((J1-mj+1.0)*(J1-mj+2.0)/2.0)
            elif (delta_J == -1.0):
                if upper_energies.has_key(mj+1.0):    # delta_Mj = +1 sigma component
                    weight = (J1-mj)*(J1-mj-1.0)
                    transitions.append(zeemanTransition(hc/(upper_energies[mj+1]-lower_energies[mj]), weight/2.0,
                    weight/4.0, mj+1, mj))
                    sigma_transitions_perp.append(upper_energies[mj+1] - lower_energies[mj])
                    sigma_transitions_para.append(upper_energies[mj+1] - lower_energies[mj])
                    sigma_weights_perp.append((J1-mj)*(J1-mj-1.0)/4.0)
                    sigma_weights_para.append((J1-mj)*(J1-mj-1.0)/2.0)
                if upper_energies.has_key(mj):        # delta_Mj = 0 Pi component
                    weight= J1**2.0 - mj**2.0
                    transitions.append(zeemanTransition(hc/(upper_energies[mj]-lower_energies[mj]), 0.0, weight, mj,
                    mj))
                    pi_transitions.append(upper_energies[mj] - lower_energies[mj])
                    pi_weights.append(J1**2.0 - mj**2.0)
                if upper_energies.has_key(mj-1.0):    # delta_Mj = -1 sigma component
                    weight = (J1+mj)*(J1+mj-1.0)
                    transitions.append(zeemanTransition(hc/(upper_energies[mj-1]-lower_energies[mj]), weight/2.0,
                    weight/4.0, mj-1, mj))
                    sigma_transitions_perp.append(upper_energies[mj-1] - lower_energies[mj])
                    sigma_transitions_para.append(upper_energies[mj-1] - lower_energies[mj])
                    sigma_weights_perp.append((J1+mj)*(J1+mj-1.0)/4.0)
                    sigma_weights_para.append((J1+mj)*(J1+mj-1.0)/2.0)

        sigma_weights_para = numpy.array(sigma_weights_para)
        sigma_weights_perp = numpy.array(sigma_weights_perp)
        pi_weights = numpy.array(pi_weights)

        scale_factor = numpy.sum(sigma_weights_perp) + numpy.sum(pi_weights) + numpy.sum(sigma_weights_para)

        sigma_weights_para /= scale_factor
        sigma_weights_perp /= scale_factor
        pi_weights /= scale_factor
        sigma_energies_para = numpy.array(sigma_transitions_para)
        sigma_energies_perp = numpy.array(sigma_transitions_perp)
        pi_energies = numpy.array(pi_transitions)

        comp_wavelengths = []
        comp_gfs = []
        comp_loggfs = []
        for transition in transitions:
            if ( (transition.weight_perp > 0) & (transition.weight_para > 0) ):
                comp_wavelengths.append(transition.wavelength)
                comp_gfs.append(transition.weight_perp*0.40183 +  transition.weight_para*0.2865)

        factor = 10.0**(self.loggf)/numpy.sum(comp_gfs)
        for gf in comp_gfs:
            comp_loggfs.append(numpy.log10(gf*factor))
        self.zeeman["FULL"] = [numpy.array(comp_wavelengths), numpy.array(comp_loggfs)]

        para_loggfs = []
        para_wavelengths = []
        perp_loggfs = []
        perp_wavelengths = []
        for line in zip(sigma_weights_para, sigma_energies_para):
           if (line[0] > 0):
               para_loggfs.append(numpy.log10(10.0**self.loggf*line[0]))
               para_wavelengths.append(hc/line[1])

        perp_weights = numpy.append(sigma_weights_perp, pi_weights)
        perp_energies = numpy.append(sigma_energies_perp, pi_energies)

        for line in zip(perp_weights, perp_energies):
            if (line[0] > 0):
                perp_loggfs.append(numpy.log10(10.0**self.loggf*line[0]))
                perp_wavelengths.append(hc/line[1])

        self.zeeman["LONG"] = [numpy.array(para_wavelengths), numpy.array(para_loggfs)]
        self.zeeman["TRANS"] = [numpy.array(perp_wavelengths),numpy.array(perp_loggfs)]

        if "mu" in kwargs:
            mu_wavelengths = []
            mu_loggfs = []
            mu = kwargs["mu"]
            for perp in zip(perp_wavelengths, perp_loggfs):
                mu_wavelengths.append(perp[0])
                if perp[0] in para_wavelengths:
                    i = para_wavelengths.index(perp[0])
                    mu_loggfs.append( numpy.log10(10.0**(perp[1])*numpy.sin(numpy.radians(mu))**2.0 +
                    10.0**(para_loggfs[i])*numpy.cos(numpy.radians(mu))**2.0))
                else:
                    mu_loggfs.append( numpy.log10(10.0**(perp[1])*numpy.sin(numpy.radians(mu))**2.0))
                   
            self.zeeman['MU_'+str(kwargs["mu"])] = [numpy.array(mu_wavelengths), numpy.array(mu_loggfs)]
 

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
                        order = numpy.argsort(self.zeeman[kwargs["zeeman"]][0])
                        for i in range(len(order)):
                            out.write('%10.3f%10s%10.3f%10.3f' % (self.zeeman[kwargs["zeeman"]][0][order[i]],
                            self.species,self.EP,self.zeeman[kwargs["zeeman"]][1][order[i]]))
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
