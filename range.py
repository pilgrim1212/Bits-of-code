for value in range(1,5):
    print(value)
##making a list with range
numbers = list(range(1,6))
print(numbers)
even_num = list(range(0,10,2))
print(even_num);
squares = []
for value in range(1,11):
    square = value ** 2
    squares.append(square)
    ##can also be done by squares.append(value**2)
print(squares)
print(min(squares))
print(max(squares))
print(sum(squares))
## can also be done with 
squares = [value**2 for value in range(1,11)]
print(squares)
### at page 99 in crash course