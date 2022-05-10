from numpy.lib.shape_base import expand_dims
import scipy.io as spio
import numpy as np
mat = spio.loadmat('training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
training_pos = 2340
test_pos = 3028
val_pos = 3343

#rain_Index = Index[:,2340]
#test_Index = Index[2340:3028], val_Index = Index[3028:3343]
np.diff(d)
GG = np.abs(np.diff(d))
#print(GG) ## finds the number of peaks based off the dofference between the absolute values,
#which indicates a spike returns length and index positions
HH = (np.abs(np.diff(d))>1.00478)
get_indexes = lambda HH, xs: [i for (y, i) in zip(xs, range(len(xs))) if HH == y]
Hi = get_indexes(True,HH)
print(Hi)
print(len(Hi))
print(HH)
print(d[Hi])
## split d[Hi] along lines for train, test and val
