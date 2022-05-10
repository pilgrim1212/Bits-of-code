#Import the Numpy library for matrix math
import numpy

## a single perceptron function
def perceptron(inputs_list, weights_list,bias):
    #converts the inputs into a numpy array
    inputs = numpy.array(inputs_list)

    #converts the weights_list into a numpy arrray
    weights = numpy.array(weights_list)

    #calculate the dot product
    summed = numpy.dot(inputs_list,weights_list)

    #Add in the bias
    summed = summed + bias

    #Calculate the output
    #N.b this is a ternary operator neat right
    output = 1 if summed > 0 else 0 

    return output

##Our main code starts here 
inputs = [1.0, 0.0]
weights =[1.0, 1.0]
bias = -1

print("Inputs: ",inputs)
print("Weights: ", weights)
print("Bias: ", bias)
print("Result: ", perceptron(inputs,weights,bias))