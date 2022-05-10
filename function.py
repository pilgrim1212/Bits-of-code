def User_greetings():
    print('Your name is ' +name +'. Your age is ' +age+ ' Years old')
name = input("Your name is: ")    
age = str(input("Your age is: "))
User_greetings()
def addition_numbers(num1, num2):
    print("new number equal to")
    return num1+num2
num1=int(input("Enter first number: "))
num2=int(input("Enter second number: "))
print(addition_numbers(num1,num2))