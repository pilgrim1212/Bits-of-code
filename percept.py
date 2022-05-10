import scipy.io as spio
import numpy as np
# import scripy.special for the sigmoid function expit()
import scipy.special,numpy, matplotlib.pyplot as plt
##import data fromn matlab file 
mat = spio.loadmat('training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
##
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
validation_data = F_test
input_val = np.asarray(d3)

target_output = data_list
validation_output = test_data_list
targets_val = np.asarray(val_data_list)

#print(training_data)
training_count = len(training_data)
validation_count = len(validation_data)



def logistic(x):
    return 1.0/(1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

LR = 1   

I_dim = 3
H_dim = 4

epoch_count = 1

#np.random.seed(1)
weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
weights_HtoO = np.random.uniform(-1, 1, H_dim)

preActivation_H = np.zeros(H_dim)
postActivation_H = np.zeros(H_dim)

#####################
#training
#####################
for epoch in range(epoch_count):
    for sample in range(training_count):
        for node in range(H_dim):
            preActivation_H[node] = np.dot(training_data[sample,:], weights_ItoH[:, node])
            postActivation_H[node] = logistic(preActivation_H[node])
            
        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)
        
        FE = postActivation_O - target_output[sample]
        
        for H_node in range(H_dim):
            S_error = FE * logistic_deriv(preActivation_O)
            gradient_HtoO = S_error * postActivation_H[H_node]
                       
            for I_node in range(I_dim):
                input_value = training_data[sample, I_node]
                gradient_ItoH = S_error * weights_HtoO[H_node] * logistic_deriv(preActivation_H[H_node]) * input_value
                
                weights_ItoH[I_node, H_node] -= LR * gradient_ItoH
                
            weights_HtoO[H_node] -= LR * gradient_HtoO

#####################
#validation
#####################            
correct_classification_count = 0
for sample in range(validation_count):
    for node in range(H_dim):
        preActivation_H[node] = np.dot(validation_data[sample,:], weights_ItoH[:, node])
        postActivation_H[node] = logistic(preActivation_H[node])
            
    preActivation_O = np.dot(postActivation_H, weights_HtoO)
    postActivation_O = logistic(preActivation_O)
        
    if postActivation_O > 0.5:
        output = 1
    else:
        output = 0     
        
    if output == validation_output[sample]:
        correct_classification_count += 1

print('Percentage of correct classifications:')
print(correct_classification_count*100/validation_count)