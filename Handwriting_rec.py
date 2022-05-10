##draw number hand writing reading files from mnist_train_100.csv
##import numpy for arrays and matplotlib for drawing of the numbers
import numpy
import matplotlib.pyplot as plt

##open the 100 training samples in read mode
data_file = open("mnist_train_100.csv",'r')

#read all the lines from the file into memory
data_list = data_file.readlines()

#close the file when we are finished with it
data_file.close()

##take the first line(data_list index 0, the first sample) and split it up based on the commas
##all_values now contain a list of (label,pixel1,pixel2,pixel3,..,pixel784)
all_values=data_list[0].split(',')

##take the long list of pixels (but not the label) and reshape them into 2D array of pixels
image_array=numpy.asfarray(all_values[1:]).reshape((28,28))

##Plot this 2D array as an image,use the grey colour map and dont interpolate
plt.imshow(image_array,cmap='Greys',interpolation='None')
plt.show()

