num1 = int(input("Enter first number:  "))
num2 = int(input("Enter the second number: "))
op = input("Enter your operator ")

if op == '+':
    print("Your addition is equal to " , num1+num2)
elif op == '-':
    print("Your subtraction is equal to ", num1-num2)
elif op == 'x':
    print("Your multiplication is equal to ", num1*num2)
elif op == '/':
    print("Your division is equal to ", num1/num2)
else:
    print("Invalid operator")