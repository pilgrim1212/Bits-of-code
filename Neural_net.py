# import scripy.special for the sigmoid function expit()
import scipy.special,numpy

#Neural network class definition
class NeuralNetwork:
    #init the neural network, this gets run everytime we make a new instance of this class
    def _init_(self,input_nodes,hidden_nodes,output_nodes,learning_rate): 
        #set the number of nodes in each input, hidden and output
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

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


    
