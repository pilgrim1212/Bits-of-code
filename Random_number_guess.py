import random

def guess(x):
    random_number = random.randint(1,x)
    guess = 0
    while guess != random_number:
        guess = int(input(f"Guess a number between 1 and {x}: "))
        if guess > random_number:
            print("sorry looks like your guess was too high")
        elif guess < random_number:
            print("Your guess was too low")
    print(f"Congratulations you sucessfully guessed {random_number}")



def computer_guess(x):
    low = 1 
    high = x
    feedback = ""
    while feedback != "c":
        guess = random.randint(low,high)
        feedback = input(f"Is {guess} too high (H), too low (L) or correct (c)").lower
        if feedback == "h":
            high = guess - 1
        elif feedback == "l":
            low = guess + 1

    print(f"Yey the computer guessed the correct number, {guess}")

computer_guess(10)