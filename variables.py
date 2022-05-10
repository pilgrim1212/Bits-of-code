name = "ada lovelace"
print(name.title()) ##makes the first letters capitalise
name_2 = "Ada lovelace"
print(name_2.title()) ## both return the same first letter capitalized
print(name.upper()) ## capitalises all the letters
print(name_2.lower()) ## lower cases all the letters
## Strings
first_name = "ada"
second_name = "lovelace" ## f allows for the variables to be substituted into the message
fullname = f" {first_name} {second_name}!"
print(fullname + "\n")## creates a new line at the end using \n
print(f"Hello, {fullname.title()}!")
print("\t this is a tab")## modes the text by a tab using \t
## stripping white space
Learning_lang = "spanish "
print(Learning_lang.rstrip())
