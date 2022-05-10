names = ["audi","bmw","cintron","ferrari","saab"]
for car in names: ## == is equal to, = is setting of value
    if car == "audi":
        print(car.upper())
    else:
        print(car.title())
for car in names:
    if car.title() =="Audi":
        print(f"This {car} is the car for me")
    else:
        print(f"this {car} is not what i want")
##checking multiple conditions using age
age_1 = 21
age_2 = input("enter value")
age_2 = int(age_2) 
if age_2 < age_1 & age_2 < 10:
    print("what are you doing here")
else:
    print("welcome")