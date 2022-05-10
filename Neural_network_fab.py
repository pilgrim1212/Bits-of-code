import numpy
import scipy.special

#Initialise the network
class Neural_Network:

    def __init__(self,input_nodes,hidden_nodes,output_nodes, learning_rate):

        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        print("Input nodes: ",self.inodes, "Hidden nodes: ", self.hnodes, "Output nodes: ", self.onodes)
        # define matrices weights times input to hidden 

        self.wih = numpy.random.normal(0.0,pow(self.inodes, -0.5),(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        #just to allow for checking of code
        print("Matrix 1 : \n", self.wih)
        print("Matrix 2 : \n", self.who)

        self.Ir = learning_rate

        ##activation fuction sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

        pass
    
    def train(self, inputs_list,targets_list):

        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.Ir * numpy.dot(output_errors*final_outputs*(1.0 - final_outputs), numpy.transpose(hidden_outputs))
        self.wih += self.Ir *numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs), numpy.transpose(inputs))

        pass

    ##query the network
    def query(self,inputs_list):

        inputs = numpy.array(inputs_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

 
