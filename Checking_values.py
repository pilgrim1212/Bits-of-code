banned_users = ["andrew", "matt","devon"]
user = "marie"
if user not in banned_users:
    print(f"{user.title()} Welcome to the server")
else:
    print(f"{user.title()} Is banned from the server")