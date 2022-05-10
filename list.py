countries = ["united kingdom", "Australia", "New Zealand", "South Africa"]
list2 = 1,2,3,4,5
countries[1] = "Germany"
print(countries[1:])
countries.append(' America')
countries.insert(2, 'cherry')
countries.extend(list2)
print(countries)