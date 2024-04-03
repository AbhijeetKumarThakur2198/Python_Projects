# Crack Zip File Password Tool

**DISCLAIMER**: PLEASE USE THIS TOOL ETHICAL PURPOSE ONLY.

## Overview
This project help user to crack password of there encrypted zip file. It don't relay on third-party module, it utilizes only pre-built modules. This project fully based on Brute Force Attack method while it really time consuming tool for cracking large password but it is good example of **Social Engineering in Cyber Security**. Currently this version only support crack password of zip only maybe in future I add additional features.

So, what is the structure of this tool first we will **import important pre-built modules** than we gone define two functions first is **get_next_series_number()** and **crack_zip_file_password()**. 

Here is the code of get_next_series_number() function:
```python
def get_next_series_number(extract_folder):
    series_numbers = []
    for folder_name in os.listdir(extract_folder):
        if os.path.isdir(os.path.join(extract_folder, folder_name)):
            series_number = folder_name.split('_')[0]
            if series_number.isdigit():
                series_numbers.append(int(series_number))
    if series_numbers:
        return max(series_numbers) + 1
    else:
        return 1
```

Usage of **get_next_series_number** to examines the folder names within a specified directory (**extract_folder**). It extracts series numbers from these folder names, considering only numeric prefixes. It then returns the next available series number by finding the highest existing number and adding 1. If no series numbers are found, it suggests starting with 1.

Here is the code of crack_zip_file_password() function:
```python
def crack_zip_file_password(zip_file, max_length=None, get_data=4):    
    if get_data == 1:    	
    	data = string.digits
    	
    elif get_data == 2:
    	data = string.ascii_letters
    	
    elif get_data == 3:
    	data = string.punctuation
    	
    elif get_data == 4:
    	data = string.digits + string.ascii_letters + string.punctuation
    
    else:
    	data = get_data

    attempt = 0    	
    password_length = 1
    loaded_zip = zipfile.ZipFile(zip_file)
    current_path = os.getcwd()
    extract_folder = os.path.join(current_path, "crackedfiles")  
    os.makedirs(extract_folder, exist_ok=True)  
    
    current_date = datetime.datetime.now()
    date = current_date.date()
    time = current_date.time()
    formatted_time = time.strftime("%I-%M-%S_%p")  
    series_number = get_next_series_number(extract_folder)

    while True:
        for word in itertools.product(data, repeat=password_length):
            password = ''.join(word)
            try:
                attempt += 1
                extract_subfolder = os.path.join(extract_folder, f"{series_number}_{zip_file}_{date}_{formatted_time}")  # Subfolder for each extraction attempt
                os.makedirs(extract_subfolder, exist_ok=True)  # Create the subfolder
                loaded_zip.extractall(path=extract_subfolder, pwd=password.encode())  # Extract files into subfolder
                print(f"\nAttempt: {attempt}, Length: {password_length}, Password: {password}\n{zip_file} successfully extracted at {extract_subfolder}")                            
                loaded_zip.close()
                return True
              
            except Exception as e:
                continue
                
        print(f"Increasing password length +1")
        password_length += 1        
                
        if max_length is not None and password_length > max_length:
            print("\nMaximum password length reached and password not found!")
            return False
```
            
Usage of **crack_zip_file_password()** is attempts to crack a encrypted ZIP file (**zip_file**) using brute force. It iterates through various combinations of characters, trying different password lengths and character sets until it successfully extracts the ZIP contents or reaches the maximum password length. It creates a new directory to store extracted files and logs each extraction attempt. The function returns True if successful, False otherwise.

## How To Use This Project
If you wish to use this project, you can do so via **Google Colab** [here](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/Crack_Zip_File_Password_Tool/CrackZipFilePassword.ipyb), or manually on your environment by copying and running the code provided below.

## Full Code
Here is the full code to crack encrypted zip file:
```python
zip_file = "YOUR_FILE.zip"  # REPLACE WITH YOUR ZIP FILE PATH
max_length = None  # SET None FOR ENDLESS LENGTH OR REPLACE WITH NUMBER LIKE 10, 20, 30 AND SO ON
get_data = 4  # SET 1 FOR DIGITS ONLY, 2 FOR STRING, 3 FOR PUNCTUATION, 4 FOR ALL OR WRITE DATA MANUALLY

# IMPORT PRE-BUILT MODULES
try:		
    import os
    import sys
    import string
    import zipfile    
    import itertools
    import datetime
except Exception as e:
	print(f"Error: {e}")
	sys.exit()

def get_next_series_number(extract_folder):
    series_numbers = []
    for folder_name in os.listdir(extract_folder):
        if os.path.isdir(os.path.join(extract_folder, folder_name)):
            series_number = folder_name.split('_')[0]
            if series_number.isdigit():
                series_numbers.append(int(series_number))
    if series_numbers:
        return max(series_numbers) + 1
    else:
        return 1

def crack_zip_file_password(zip_file, max_length=None, get_data=4):    
    if get_data == 1:    	
    	data = string.digits
    	
    elif get_data == 2:
    	data = string.ascii_letters
    	
    elif get_data == 3:
    	data = string.punctuation
    	
    elif get_data == 4:
    	data = string.digits + string.ascii_letters + string.punctuation
    
    else:
    	data = get_data

    attempt = 0    	
    password_length = 1
    loaded_zip = zipfile.ZipFile(zip_file)
    current_path = os.getcwd()
    extract_folder = os.path.join(current_path, "crackedfiles")  
    os.makedirs(extract_folder, exist_ok=True)  
    
    current_date = datetime.datetime.now()
    date = current_date.date()
    time = current_date.time()
    formatted_time = time.strftime("%I-%M-%S_%p")  
    series_number = get_next_series_number(extract_folder)

    while True:
        for word in itertools.product(data, repeat=password_length):
            password = ''.join(word)
            try:
                attempt += 1
                extract_subfolder = os.path.join(extract_folder, f"{series_number}_{zip_file}_{date}_{formatted_time}")  # Subfolder for each extraction attempt
                os.makedirs(extract_subfolder, exist_ok=True)  # Create the subfolder
                loaded_zip.extractall(path=extract_subfolder, pwd=password.encode())  # Extract files into subfolder
                print(f"\nAttempt: {attempt}, Length: {password_length}, Password: {password}\n{zip_file} successfully extracted at {extract_subfolder}")                            
                loaded_zip.close()
                return True
              
            except Exception as e:
                continue
                
        print(f"Increasing password length +1")
        password_length += 1        
                
        if max_length is not None and password_length > max_length:
            print("\nMaximum password length reached and password not found!")
            return False

crack_zip_file_password(zip_file=zip_file, max_length=max_length, get_data=get_data)
```

## How To Report Problem
If you encounter any errors or bugs in the code, please create an issue Here.

## License
This project is under the [MIT License](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) - see the [LICENSE.md](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) file for details.
