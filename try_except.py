try:
    x= int(input("Enter a value"))
    print(x)
except ValueError:
    print("Invalid input place an integer")
else:
    print("Nothing went wrong")
finally:
    print("text appears regardless")