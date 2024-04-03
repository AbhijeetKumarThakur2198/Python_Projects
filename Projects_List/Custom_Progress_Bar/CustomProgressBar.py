def custom_progress_bar(iterable, status=False, symbol="#", color=False):
    bar_length = 20
    total = len(iterable)
    previous_block = 0
    counter = 0
    error_bar = ""

    try:
        for item in iterable:
            counter += 1
            progress = counter / total
            block = int(round(bar_length * progress))
            previous_block = block

            if color:
                green_symbol = "\033[92m" + symbol + "\033[0m"
                progress_bar = "[" + green_symbol * block + "-" * (bar_length - block) + "]"
            else:
                progress_bar = "[" + symbol * block + "-" * (bar_length - block) + "]"

            print(f"\r{progress_bar} {progress * 100:.0f}%", end='', flush=True)
            yield item
        print('\n')

        if status:
            if color:
                print("\033[92mProcess completed!\033[0m\n")
            else:
                print("Process completed!\n")

    except Exception as e:
        if color==True:
            red_symbol = "\033[91m" + symbol + "\033[0m"
            error_bar = "[" + red_symbol * previous_block + "-" * (bar_length - previous_block) + "]"
        else:
        	error_bar = "[" + symbol * previous_block + "-" * (bar_length - previous_block) + "]"
        	
        print(f"\r{error_bar}", end='', flush=True)
        print("\n")
       
        if status:
            if color:        	
                print(f"\033[91mError: {e}\033[0m\n")
            else:
            	print(f"Error: {e}\n")
