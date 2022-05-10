toppings = ["pepperoni","mushrooms", "cheese"]
if "pepperoni" in toppings:
    print(f"adding {toppings[0]}")
if "mushrooms" in toppings:
    print(f"adding {toppings[1]}")
if "cheese" in toppings:
    print(f"adding {toppings[2]}")
print("\n Pizza is ready")
##also achieved by
for topping in toppings:
    if topping == "mushrooms":
        print(f"we are out of {topping}")
    else:
        print(f"Adding {topping}. !")
print("pizza ready")
## using two lists
requested_topping =  ["green pepper","chilli"]
for topping in toppings:
    if requested_topping in toppings:
        print(f"adding {requested_topping}")
    else:
        print("apologies we dont have that topping")