import numpy

class Synthetic_Stokes( object ):
    def __init__(self, name):
        self.name = name
        self.angles = []
        self.wl = dict()
        self.I = dict()
        self.Q = dict()
        self.U = dict()
        self.V = dict()
        self.C = dict()

    def set_wl(self, wl, angle):
        if not(angle in self.angles):
            self.angles.append(float(angle))
        self.wl[angle] = numpy.array(wl)

    def set_I(self, I, angle):
        if not(angle in self.angles):
            self.angles.append(float(angle))
        self.I[angle] = numpy.array(I)

    def set_Q(self, Q, angle):
        if not(angle in self.angles):
            self.angles.append(float(angle))
        self.Q[angle] = numpy.array(Q)

    def set_U(self, U, angle):
        if not(angle in self.angles):
            self.angles.append(float(angle))
        self.U[angle] = numpy.array(U)

    def set_V(self, V, angle):
        if not(angle in self.angles):
            self.angles.append(float(angle))
        self.V[angle] = numpy.array(V)

    def set_C(self, C, angle):
        if not(angle in self.angles):
            self.angles.append(float(angle))
        self.C[angle] = numpy.array(C)

    def get_angles(self):
        return self.angles

    def get_wl(self, angle):
        return self.wl[angle]

    def get_I(self, angle):
        return self.I[angle]

    def get_Q(self, angle):
        return self.Q[angle]

    def get_U(self, angle):
        return self.U[angle]

    def get_V(self, angle):
        return self.V[angle]

    def get_C(self, angle):
        return self.C[angle]

    def read_prf(self, df):
        prf = open(df, 'r')
        l = prf.readline()
        l = prf.readline()
        l = prf.readline()
        n_angles = int(l.split()[0])
        i = 0
        while i < n_angles:
            line = prf.readline()
            l = line.split()
            i = int(l[0])
            mu = float(l[1])
            l = prf.readline().split()
            wl_start = float(l[0])
            c1 = float(l[1])
            wl_stop = float(l[2])
            c2 = float(l[3])
            slope = (c2-c1)/(wl_stop-wl_start)
            num_wl_pts = int(prf.readline().split()[0])
            wl = []
            C = []
            while len(wl) < num_wl_pts:
                wls = prf.readline().split()
                for x in wls:
                    wl.append(float(x))
                    C.append(c1+(wl[-1]-wl_start)*slope)
            self.set_wl(wl, mu)
            self.set_C(C, mu)
            I = []
            while len(I) < num_wl_pts:
                Is = prf.readline().split()
                for x in Is:
                    I.append(float(x))
            self.set_I(I, mu)
            Q = []
            while len(Q) < num_wl_pts:
                Qs = prf.readline().split()
                for x in Qs:
                    Q.append(float(x))
            self.set_Q(Q, mu)
            U = []
            while len(U) < num_wl_pts:
                Us = prf.readline().split()
                for x in Us:
                    U.append(float(x))
            self.set_U(U, mu)
            V = []
            while len(V) < num_wl_pts:
                Vs = prf.readline().split()
                for x in Vs:
                    V.append(float(x))
            self.set_V(V, mu)
