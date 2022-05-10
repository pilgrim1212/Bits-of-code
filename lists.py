#lists are ways of storing elements
squares = [1,4,9,16,25]
#just like strings they can be indexed(from 0) and sliced
one = squares(0)
end = squares(-1)
#notice how the slice operation returns a new list containing the elements
new_squares = squares[0:3]
#lists may be concatinated
squares = squares + [36, 49, 64, 81]
#unlike strings, lists are mutatble you can change the individual elements
squares(0)= 1**2
#len also works on lists
squarelen = len(squares)
print(squarelen)