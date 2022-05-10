# import scripy.special for the sigmoid function expit()
import scipy.special,numpy, matplotlib.pyplot as plt

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

##open the 100 training samples in read mode
data_file = open("mnist_train_100.csv",'r')
#read all the lines from the file into memory
data_list = data_file.readlines()
#close the file when we are finished with it
data_file.close()
len(data_list)
Test_data_file = open("mnist_test.csv",'r')
test_data_list = Test_data_file.readlines()
Test_data_file.close()

##take the first line(data_list index 0, the first sample) and split it up based on the commas
##all_values now contain a list of (label,pixel1,pixel2,pixel3,..,pixel784)
all_values=data_list[0].split(',')

##take the long list of pixels (but not the label) and reshape them into 2D array of pixels
image_array=numpy.asfarray(all_values[1:]).reshape((28,28))

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

        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/ 255.0*0.99)+0.01

        targets = numpy.zeros(output_nodes) + 0.01
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