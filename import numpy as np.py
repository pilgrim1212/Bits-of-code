import numpy as np 
import scipy.io as spio
##import data from matlab file store in variable d,Index and Class
mat = spio.loadmat('training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
print(Class)
print(Index)
##Find spikes
##make an array of the values of D then find the position vectors of the spikes
np.diff(d)
GG = np.abs(np.diff(d))
#print(GG) check see if GG is correct

#print(GG) ## finds the number of peaks based off the dofference between the absolute values,
#which indicates a spike returns length and index positions num spikes should be 3343
HH = (np.abs(np.diff(d))>1.00478)# values returns 3343 positions
get_indexes = lambda HH, xs: [i for (y, i) in zip(xs, range(len(xs))) if HH == y]
Hi = get_indexes(True,HH) ## indexes for spikes in D
#print(Hi)
#print(len(Hi))
#print(HH)
#print(d[Hi])
## split d[Hi] along lines for train, test and val
split_list = [2340,3028,3343]
F = d[Hi] ## slit D into training test and validation
F_total = [F[i : j] for i, j in zip([0] + split_list, split_list + [None])]
F_train = F_total[0] ##store as these variables
F_test = F_total[1] 
F__val = F_total[2] 
Len_train = [*range(1,len(F_train))]##for when building 2d arrays with variables
Len_test = [*range(1,len(F_test))]
Len_val = [*range(1,len(F__val))]
##split both the class and index variables into train, test and validation
Class_total = [Class[i : j] for i, j in zip([0] + split_list, split_list + [None])]
Class_train = Class_total[0]
Class_test = Class_total[1]
Class_val = Class_total[2]
#print(Class_train) #check if works
Index_total = [Index[i : j] for i, j in zip([0] + split_list, split_list + [None])]
Index_train = Index_total[0]
Index_test = Index_total[1]
Index_val = Index_total[2]
#check to see if passes through
#print(Index_test)
##assemble arrays for index class and d to pass through
output1 = [[0]*2 for i in range(len(Index_train))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Index_train, Class_train):
    output1[k][0]=i
    output1[k][1]=j
    k+=1

#data_list = np.asarray(data_list)
output2 = [[0]*2 for i in range(len(Index_test))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Index_test, Class_test):
    output2[k][0]=i
    output2[k][1]=j
    k+=1

output3 = [[0]*2 for i in range(len(Index_val))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Index_val, Class_val):
    output3[k][0]=i
    output3[k][1]=j
    k+=1
## now the D values for 2d arrays
input1 = [[0]*2 for i in range(len(F_train))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_train, F_train):
    input1[k][0]=i
    input1[k][1]=j
    k+=1

input2 = [[0]*2 for i in range(len(F_test))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_test, F_test):
    input2[k][0]=i
    input2[k][1]=j
    k+=1
input3 = [[0]*2 for i in range(len(F_test))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_test, F_test):
    input3[k][0]=i
    input3[k][1]=j
    k+=1
##now have the inputs and outputs for the three data sets
## no the d values
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
Len_test = [*range(1,len(F_test))]
mat3 = [[0]*2 for i in range(len(F_train))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_train, F_train):
    mat3[k][0]=i
    mat3[k][1]=j
    k+=1

mat4 = [[0]*2 for i in range(len(F_test))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_test, F_test):
    mat4[k][0]=i
    mat4[k][1]=j
    k+=1
## assign input and output
X = mat3 # d values and spike num
Y = mat1  # index and class values
#print(X) # check if passed through correctly
#

