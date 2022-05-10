from numpy.lib.shape_base import expand_dims
import scipy.io as spio
import numpy as np
from array import *
mat = spio.loadmat('training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
##make an array of the values of D then find the position vectors of the spikes
##make an array of the values of D then find the position vectors of the spikes
np.diff(d)
GG = np.abs(np.diff(d))
#print(GG)
#print(len(GG>0.2))
np.diff(d)
GG = np.abs(np.diff(d))
#print(GG) ## finds the number of peaks based off the dofference between the absolute values,
#which indicates a spike returns length and index positions
HH = (np.abs(np.diff(d))>1.00478)
get_indexes = lambda HH, xs: [i for (y, i) in zip(xs, range(len(xs))) if HH == y]
Hi = get_indexes(True,HH)
#print(Hi)
#print(len(Hi))
#print(HH)
#print(d[Hi])
## split d[Hi] along lines for train, test and val
split_list = [2340,3028,3343]
F = d[Hi]
F_total = [F[i : j] for i, j in zip([0] + split_list, split_list + [None])]
F_train = F_total[0] 
F_test = F_total[1] 
F__val = F_total[2] 
#split_list_2 = [2034,3028]
#split_list_3 = [3028,3343]
Class_total = [Class[i : j] for i, j in zip([0] + split_list, split_list + [None])]
Class_train = Class_total[0]
Class_test = Class_total[1]
Class_val = Class_total[2]

Index_total = [Index[i : j] for i, j in zip([0] + split_list, split_list + [None])]
Index_train = Index_total[0]
Index_test = Index_total[1]
Index_val = Index_total[2]
#print(d1); print(len(d1))
#print(Index_test)
mat1 = [[0]*2 for i in range(len(Index_train))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Index_train, Class_train):
    mat1[k][0]=i
    mat1[k][1]=j
    k+=1
data_list = mat1
#data_list = np.asarray(data_list)
mat2 = [[0]*2 for i in range(len(Index_test))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Index_test, Class_test):
    mat2[k][0]=i
    mat2[k][1]=j
    k+=1
test_data_list = mat2

val_data_list = Index_val,Class_val,F__val
#print(data_list)
##
In_tr = F_train[-1]
in_te = F_test[-1]
int_val = F__val[-1]
#print(In_tr,in_te,int_val)
d_pos1 = get_indexes(In_tr,d)
d_pos2 = get_indexes(in_te,d)
d_pos3 = get_indexes(int_val,d)
#print(d_pos1[0],d_pos2[0],d_pos3[0])
split_ld = [d_pos1[0],d_pos2[0],d_pos3[0]]
d_tots = [d[i : j] for i, j in zip([0] + split_ld, split_ld + [None])]
d1 = d_tots[0]
#print(d1)
d2 = d_tots[1]
d3 = d_tots[2]
Len_train = [*range(1,len(F_train))]
mat3 = [[0]*2 for i in range(len(F_train))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_train, F_train):
    mat3[k][0]=i
    mat3[k][1]=j
    k+=1

mat4 = [[0]*2 for i in range(len(F_test))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_train, F_test):
    mat4[k][0]=i
    mat4[k][1]=j
    k+=1
#Data_type = object
  # This cause Value error
#np_array = numpy.array(F_train, dtype=Data_type)
#print(np_array)
training_data = mat3
validation_data = mat4
input_val = np.asarray(d3)
print(mat3)
target_output = data_list
validation_output = test_data_list
targets_val = np.asarray(val_data_list)

#print(training_data)
training_count = len(training_data)
validation_count = len(validation_data)