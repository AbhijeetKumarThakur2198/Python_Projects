#####|| Configuration ||#####
zip_file = "YOUR_FILE.zip"  # REPLACE WITH YOUR ZIP FILE PATH
max_length = None  # SET None FOR ENDLESS LENGTH OR REPLACE WITH NUMBER LIKE 10, 20, 30 ETC
get_data = 4  # SET 1 FOR DIGITS ONLY, 2 FOR STRING, 3 FOR PUNCTUATION, 4 FOR ALL OR WRITE DATA MANUALLY
######################

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

crack_zip_file_password(zip_file=zip_file, max_length=max_length, get_data=get_data)
