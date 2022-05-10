#import the matplotlib pyplot library
#it has a very long name, so import it as the name plt
import matplotlib.pyplot as plt

#make a new plot (XKCD style)
fig = plt.xkcd()

#add points as scatters -scatter(x,y,size,colour)
#zorder determines the drawing order, set to 3 to make
#grid lines appear behind the scatter points
plt.scatter(0,0,s=50,color="green",zorder=3)
plt.scatter(0,1,s=50,color="green",zorder=3)
plt.scatter(1,0,s=50,color="green",zorder=3)
plt.scatter(1,1,s=50,color="red",zorder=3)
#plot line 
plt.plot([2,-0.5],[-0.5,2],"g--")


#set the axis limits
plt.xlim(-0.5,2)
plt.ylim(-0.5,2)

#Label the plot
plt.xlabel("input1")
plt.ylabel("input2")
plt.title("State space of Input Vector")

#turn on grid lines
plt.grid(True,linewidth=1,linestyle=':')

#Autosize (stops the labels being cut off)
plt.tight_layout()

#Show the plot
plt.show()