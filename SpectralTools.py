import scipy.signal
import scipy.interpolate
import numpy

def resample(x, y, R):
    ''' This routine convolves a given spectrum to a resolution R'''

    old_length = len(x)

    subsample = 8.0
    stepsize = (x/(R))/4.0

    newx = [x[0]]
    xp = x[0]
    for i in range(1,old_length):
        while (xp < x[i]):
            xp += stepsize[i]
            newx.append(xp)
    if newx[-1] > max(x):
       junk = newx.pop()

    f = scipy.interpolate.interpolate.interp1d(x, y, bounds_error=False)
    newy = f(newx)
    const = numpy.ones(len(newx))

    xk = numpy.array(range(4*subsample))
    yk = numpy.exp(-(xk-2*subsample)**2.0/(subsample**2/(4*numpy.log(2.0))))
    
    result = scipy.signal.convolve(newy, yk, mode ='same')
    normal = scipy.signal.convolve(const, yk, mode = 'same')

    return newx, result/normal
