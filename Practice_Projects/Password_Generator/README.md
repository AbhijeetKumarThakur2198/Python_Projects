# MY PRACTICE PROJECT 3

This Python script provides a simple command-line interface to generate and save secure passwords. It utilizes the `random` and `string` modules for password generation.

## Usage

1. Run the script.
2. Enter the desired length for your password when prompted.
3. The generated password will be saved to the default file "saved_passwords.txt".
4. Optionally, you can modify the filename by providing it as an argument in the `save_password` function.

## Code Structure

- `generate_password(length=12)`: Function to generate a password with a default length of 12 characters.
- `save_password(password, filename="saved_passwords.txt")`: Function to save a password to a file. The default filename is "saved_passwords.txt".
- `password_length = int(input("Enter the desired length of the password: "))`: Prompts the user for the desired password length.
- Generates and saves a password using the functions defined above.

## Full code:

```python
# MY PRACTICE PROJECT 3
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
```
If you find this project interesting, feel free to use and modify it for integration into more complex programs.

