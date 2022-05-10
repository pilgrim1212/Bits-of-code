import numpy
##load the Mnist test samples csv file into a list
test_data_file = open("mnist_test_10.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#scorecard list for how well the network preforms,initially empty
scorecard=[]

##loop through for all the records in the dataset
for records in test_data_list:
    #Split the values according to the commas
    all_values = test_data_list.split(',')
    #the correct label is the first value
    correct_label = int(all_values[0])
    print(correct_label,"Correct label")
    #scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01
    #query the network
    outputs = n.query(inputs)
    #the index of the highest value output corresponds to the label
    label = numpy.argmax(outputs)
    print(label,"Network Label")
    ##append either a 1 or 0 to the score card list
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

#calculate the preformance score, the fraction of the correct answers
scorecard_array = numpy.asarray(scorecard)
print("Preformance= ",(scorecard_array.sum()/scorecard_array.size)*100,"100")
