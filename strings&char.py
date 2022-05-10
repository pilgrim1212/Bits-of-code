#strings can be contained in either single or double quotes
name = "bob"
Name = 'bobby'
#but when using a ' within sentence becareful and put a \ before
##strings may be concatinated 
both_names = name +' ' + Name # ' ' for a space inbetween
sent = 'It\'s garries'
#strings can be indexed with the first character having the index of zero
# there is no character type, a character is simply a string of length 1
firstletter = both_names[1]
# range of the index can be done
some_letters= both_names[0:3]
##you cant change modify a string, but you can overwrite it or make a new one
# you can check the length with len() 
len(both_names)
