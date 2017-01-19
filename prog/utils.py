import time
import numpy as np


def timeit(method):

    def repr_arg(x):
        if isinstance(x, np.ndarray):
            return 'np.array %s' % str(x.shape)
        return str(x)

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        pargs = list(map(repr_arg, list(args) + list(kw.values())))

        print('{!r} ({!r}) {:.2f} sec'.format(method.__name__, pargs, te-ts))
        return result

    return timed
