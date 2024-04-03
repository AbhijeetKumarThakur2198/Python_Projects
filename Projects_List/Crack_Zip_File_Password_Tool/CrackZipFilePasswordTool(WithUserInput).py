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
                extract_subfolder = os.path.join(extract_folder, f"{series_number}_{zip_file}_{date}_{formatted_time}")  
                os.makedirs(extract_subfolder, exist_ok=True)  
                loaded_zip.extractall(path=extract_subfolder, pwd=password.encode())  
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

while True:
	print("Select one of these:\n1) Crack zip file password\n2) Exit the program")
	ask = int(input("Enter your choice: "))
	if ask == 1:		
		get_zip_file_path = input("Enter your encrypted zip file path: ")
		print("Select one of these:\n1) Increase length until password found\n2) Set length mannuly\n3) Help\n")
		ask_max_len = int(input("Enter your choice: "))
		
		if ask_max_len ==3:
			print("First option set max length of password to crack until the password will crack, second option will help you to set max length of password means if you set max length 5 then the prgram auto stop when it reach max length.\n\nSelect one of these:\n1) Increase length until password found\n2) Set length mannuly")
			ask_max_len = int(input("Enter your choice: "))
		
		elif ask_max_len == 1:
			max_length = None
			
		elif ask_max_len == 2:
			get_max_len = int(input("Enter max length: "))		
			max_length = get_max_len
					
		else:
			print(f"Wrong input got {ask_max_len}")

		print("Select one of these:\n1) Digits\n2) String\n3) Special Character\n4) All\n5) Custom")
		ask_data = int(input("Enter your choice: "))
		
		if ask_data == 1:
			get_data = 1
		elif ask_data == 2:
			get_data = 2
		elif ask_data == 3:
			get_data = 3
		elif ask_data == 4:
			get_data = 4
		elif ask_data == 5:
			print("Select one option:\n1) Use txt file\n2) Type manually")
			ask_data = int(input("Enter your choice: "))
			if ask_data == 1:
				ask_file = input("Enter path of your data.txt file: ")
				with open(ask_file, "r") as file:
					get_data = file.read()
			elif ask_data == 2:				
				get_data = input("Write data: ")
	
		crack_zip_file_password(get_zip_file_path, max_length=max_length, get_data=get_data)
				
	elif ask == 2:
		print("Program exited")
		break
