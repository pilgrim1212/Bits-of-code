import random

def play():
    user = input("'r' for rock, 's' for scissors and 'p' for paper:")
    computer = random.choice(['r','s','p'])

    if user == computer:
        return "It\'s a tie"
    
    # r > s, s>p, p>r
    if is_win(user, computer):
        return "Congratulations You WInn!!"
    
    return "You Looooose!! Play again "


def is_win(player, opponent):
    #return true if the player wins
    if (player =='r' and opponent == 's') or (player == 's' and opponent == 'p') or (player == 'p' and opponent == 'r'):
        return True 
print(play())