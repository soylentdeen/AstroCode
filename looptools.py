import scipy
import pyfits
import numpy
import re

def measureResidualGradients(dfname):
    data = pyfits.getdata(dfname)
    grad = data.field(4)
    xvals = numpy.arange(grad.shape[1]/2)
    yvals = xvals +1
    
    mean_x = []
    mean_y = []
    std_x = []
    std_y = []

    for g in grad:
        mean_x.append(numpy.mean(g[xvals]))
        mean_y.append(numpy.mean(g[yvals]))
        std_x.append(numpy.std(g[xvals]))
        std_y.append(numpy.std(g[yvals]))

    return mean_x, mean_y, std_x, std_y

def readLoopFile(dfname):
    loop = pyfits.getdata(dfname)
    return loop

def readRTCoutput(dfname):
    values = [[],[],[],[],[]]
    with open(dfname) as f:
        a = 0
        for line in f:
            if (line[0] == 'H'):
                #l = line.split()
                #fnum = int(l[-1][0:-1])
                #try:
                #    if fnum != values[0][-1]:
                #        values[0].append(fnum)
                #except:
                #    values[0].append(fnum)
                if re.search("Buffer", line):
                    a = 1
                else:
                    a = 2
            elif (line[0] == 'A'):
                #l = line.split()
                #fnum = int(l[-1][0:-1])
                #try:
                #    if fnum != values[0][-1]:
                #        values[0].append(fnum)
                #except:
                #    values[0].append(fnum)
                if re.search("Buffer", line):
                    a = 3
                else:
                    a = 4
            else:
                vals = []
                l = line[1:-3].split()
                for num in l:
                    vals.append(int(num, 16))
                values[a].append(numpy.array(vals))
                if a == 1:
                    values[0].append(vals[0])
    return values
