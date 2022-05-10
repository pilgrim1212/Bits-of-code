#Class template for a dog object
class Dog:
    #initialisation method, gets run whenever we create a new Dog
    #the self just allows this function to reference variables relevent to this particular Dog
    def _init_(self,name,hungerLevel):
        self.name = name
        self.hungerLevel = hungerLevel
    
    #Query the status of the Dog
    def status(self):
        print("Dog is called ", self.name)
        print("Dog hunger level is ", self.hungerLevel)
        pass

    #Set the hunger level of the Dog
    def setHungerLevel(self,hungerLevel):
        set.hungerLevel = hungerLevel
        pass

    #Dogs can bark
    def bark(self):
        print("Woof!")
        pass

#Create two dog objects
#Note that we dont need to include the self from the parameter
lassie = Dog("Lassie","Mild")
yoda = Dog("Yoda","Ravenous")

#Check on Yoda and lassie
yoda.status()
lassie.status()

#Get Lassie to bark
lassie.bark()

#Feed Yoda
yoda.setHungerLevel("Full")
yoda.status()
