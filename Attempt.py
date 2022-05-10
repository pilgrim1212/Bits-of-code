import numpy as np
from scipy.stats import truncnorm

                     
image_size = 28 # width and length
num_of_labels = 10 #0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
train_data = np.loadtxt("mnist_train_100.csv", delimiter=",")
test_data = np.loadtxt("mnist_test_10.csv", delimiter=",")
                       
                       
fac = 255  *0.99 + 0.01
train_imgs = np.asfarray(train_data[:, 1:]) / fac
test_imgs = np.asfarray(test_data[:, 1:]) / fac
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(10)
for label in range(10):
    one_hot = (lr==label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)
    
lr = np.arange(num_of_labels)
# Turns labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)
# We don't want zeroes and ones
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99


def sigmoid(x):
    return 1 / (1 + np.e ** -x)
    
activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)

#Neural network class definition
class NeuralNetwork:
    #init the neural network, this gets run everytime we make a new instance of this class
    def _init_(self,input_nodes,hidden_nodes,output_nodes,learning_rate): 
        #set the number of nodes in each input, hidden and output
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        #weight matrices, with (output->hidden) and who (hidden->output)
        self.wih = np.random.normal(0.0,pow(self.h_nodes, -0.5),(self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0,pow(self.o_nodes,-0.5),(self.o_nodes,self.h_nodes))

        #Set the learning rate
        self.Ir = learning_rate

        

    #Train the neural network using back propagation of errors
    def train(self,inputs_list,targets_list):
        #Convert inputs into 2d arrays
        inputs_array = np.array(inputs_list, ndmin=2).T
        targets_array = np.array(targets_list, ndmin=2).T
                
        #Calculate the signals into hidden layer
        hidden_inputs = np.dot(self.wih,inputs_array)

        #calculate the signals emerging from the hidden layer
        hidden_outputs = activation_function(hidden_inputs)

        #Calculate the signals into the final output layer
        final_inputs = np.dot(self.who,hidden_outputs)

        #Calculate the signals emerging from the final output layer
        final_outputs = activation_function(final_inputs)

        #Current error is (target - actual)
        output_errors= targets_array - final_outputs

        #Hidden layer errors are the output errors, split by the weights, recombined at the hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        #update the weights for the links between the hidden and output layers
        self.who += self.Ir*np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layers
        self.wih += self.Ir*np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs_array))

    
    #Query the network
    def query(self,inputs_list):
        #Convert the inputs list into a 2d array
        inputs_array = np.array(inputs_list,ndmin=2).T

        #calculate the signals into the hidden layer
        hidden_inputs = np.dot(self.wih,inputs_array)

        #Calculate output from the hidden layer
        hidden_outputs = activation_function(hidden_inputs)

        #calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)

        #calculate outputsfrom the final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs     

    def run(self, input_vector):
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.wih, 
                               input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.who, 
                               output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector
            
epochs = 10
            
ANN = NeuralNetwork(input_nodes = 784, 
                    output_nodes = 10, 
                    hidden_nodes = 100,
                    learning_rate = 0.15)       

for epoch in range(epochs):  
    print("epoch: ", epoch)
    for i in range(len(train_imgs)):
        ANN.train(train_imgs[i], train_labels_one_hot[i])
        
f = open("betterNN-3.csv", 'w')
f.write("id,solution" + "\n")
for i in range(10000):
    res = ANN.run(test_imgs[i])
    f.write(str(i + 1) + "," + str(np.argmax(res)) + "\n")