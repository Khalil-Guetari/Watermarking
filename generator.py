import numpy as np
from pylfsr import LFSR


fpoly = [13,4,3,1]

L = LFSR(fpoly=fpoly, initstate='random', verbose=True)
L.runKCycle(4096)
seq = L.seq
print(len(seq))
