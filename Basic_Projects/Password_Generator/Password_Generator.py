# IMPORT RANDOM AND STRING MODULES FOR PASSWORD GENERATION
import random
import string

# FUNCTION TO GENERATE A PASSWORD WITH DEFAULT LENGTH OF 12
def generate_password(length=12):
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for _ in range(length))
    return password

# FUNCTION TO SAVE PASSWORD TO A FILE (DEFAULT: "saved_passwords.txt")
def save_password(password, filename="saved_passwords.txt"):
    with open(filename, "a") as file:
        file.write(password + "\n")

# PROMPT USER FOR DESIRED PASSWORD LENGTH
password_length = int(input("Enter the desired length of the password: "))

# GENERATE AND SAVE A PASSWORD
new_password = generate_password(password_length)
save_password(new_password)

# EXAMPLE: GENERATE AND SAVE MORE PASSWORDS
for _ in range(5):
    additional_password = generate_password(password_length)
    save_password(additional_password)
