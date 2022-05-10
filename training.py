import numpy
##load the training data frin tge mnist_train_100.csv into a list
training_data_file = open("mnist_train_100.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the nheural network on each training sample
for record in training_data_list:
    #split the record accoring to each comma
    all_values = record.split(',')
    # scale and shift the inputs form 0 .. 255 to 0.01 .. 1
    inputs =(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    #Create the target output values (all 0.01 except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes)+0.01
    #All_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    #train the network
    n.train(inputs,targets)
pass

