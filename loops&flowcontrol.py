#Loops are great ways of repeating code
a = 0
b = 1
#important that the contents of the loop are contained within the indent (single tab)
#Loops can be nested you just have to keep indenting
while b<10:
    print(b)
    a = b
    b = a + b
#we can also use for loops
#as in when we need to run through a list
words = ['cat', 'window', 'dog']
for word in words:
    print(words)
# if we wish to loop a set number of times we can make a range of intergers using range()
#range(10) produces [0,1,2,3,4,5,6,7,8,9]
for i in range(10):
    print(i)
#we can preform the conditional check using the if statement
# the else and elif are optional not the indentation
pi = int(input("input a value of pi"))
if pi < 3:
    pi = 3.141
    print("Who ate all the pi!")
elif pi > 4:
    print("Too much pi")
else:
    pi = 0
    print("no pi")