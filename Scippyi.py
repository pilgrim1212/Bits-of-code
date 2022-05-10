# import scripy.special for the sigmoid function expit()
import scipy.special,numpy, matplotlib.pyplot as plt
import numpy as np
#Neural network class definition
class NeuralNetwork:
    #init the neural network, this gets run everytime we make a new instance of this class
    def __init__(self,input_nodes,hidden_nodes,outputnodes,learning_rate): 
        #set the number of nodes in each input, hidden and output
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = outputnodes

        #weight matrices, with (output->hidden) and who (hidden->output)
        self.wih = numpy.random.normal(0.0,pow(self.h_nodes, -0.5),(self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0,pow(self.o_nodes,-0.5),(self.o_nodes,self.h_nodes))

        #Set the learning rate
        self.Ir = learning_rate

        #set the activation function, the logistic sigmoid
        self.activation_function = lambda x:scipy.special.expit(x)

    #Train the neural network using back propagation of errors
    def train(self,inputs_list,targets_list):
        #Convert inputs into 2d arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T
                
        #Calculate the signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs_array)

        #calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #Calculate the signals into the final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)

        #Calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        #Current error is (target - actual)
        output_errors= targets_array - final_outputs

        #Hidden layer errors are the output errors, split by the weights, recombined at the hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #update the weights for the links between the hidden and output layers
        self.who += self.Ir*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layers
        self.wih += self.Ir*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs_array))

    #Query the network
    def query(self,inputs_list):
        #Convert the inputs list into a 2d array
        inputs_array = numpy.array(inputs_list,ndmin=2).T

        #calculate the signals into the hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs_array)

        #Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calculate outputsfrom the final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs



import scipy.io as spio
mat = spio.loadmat('training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
print(Class)
print(Index)

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
Len_train = [*range(1,len(F_train))]
Len_test = [*range(1,len(F_test))]
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
for i, j in zip(Len_train,Index_train):
    mat1[k][0]=i
    mat1[k][1]=j
    k+=1
data_list = mat1
#data_list = np.asarray(data_list)
mat2 = [[0]*2 for i in range(len(Index_test))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_test,Index_test):
    mat2[k][0]=i
    mat2[k][1]=j
    k+=1
test_data_list = mat2
##Class mat
cla1 = [[0]*2 for i in range(len(Class_train))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_train,Class_train):
    cla1[k][0]=i
    cla1[k][1]=j
    k+=1
data_list = cla1

cla2 = [[0]*2 for i in range(len(Class_test))]## makes an array of 2 columns, rows equal to len index test
k=0
for i, j in zip(Len_train,Class_test):
    cla2[k][0]=i
    cla2[k][1]=j
    k+=1
data_list = cla2
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
## assign i
#Data_type = object
  # This cause Value error
#np_array = numpy.array(F_train, dtype=Data_type)
#print(np_array)
training_data = mat3
validation_data = mat4

print(mat3)
data_list = mat3
test_data_list = mat4
#targets_val = np.asarray(val_data_list)
target_output = mat1
#print(training_data)
training_count = len(training_data)
validation_count = len(validation_data)
##
##
##


##take the first line(data_list index 0, the first sample) and split it up based on the commas
##all_values now contain a list of (label,pixel1,pixel2,pixel3,..,pixel784)
all_values=data_list[0]

##take the long list of pixels (but not the label) and reshape them into 2D array of pixels
#image_array=numpy.asfarray(all_values[1:])

##Plot this 2D array as an image,use the grey colour map and dont interpolate
#plt.imshow(image_array,cmap='Greys',interpolation='None')
#plt.show()

##training of the neural network
# #epoch is the number of times it circles through the dataset
# need to fix unable to take arguments into neural network
n = NeuralNetwork(input_nodes=784,hidden_nodes=100,outputnodes=10,learning_rate =0.5)
epochs = 9
output_nodes =10
for e in range(epochs):
    print("Epoch: ", e +1)

    for record in data_list:

        all_values = record
        inputs = (numpy.asfarray(all_values[1:])/ 255.0*0.99)+0.01

        targets = target_output + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)

# test the neural network
scorecard = []

for record in test_data_list:
    all_values = record.split(',')

    correct_label= int(all_values[0])

    inputs = (numpy.asfarray(all_values[1:]) / 255.0 *0.99) + 0.01

    outputs = n.query(inputs)

    label = numpy.argmax(outputs)

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asfarray(scorecard)
print("Preformance: ", scorecard_array.sum() / scorecard_array.size)
#image_array=numpy.asfarray(scorecard_array.sum()).reshape((28,28))
#plt.imshow(image_array,cmap='Greys',interpolation='None')
#plt.show()