# Custom Progress Bar

## Overview
In this project, I crafted a simplistic progress bar module without relying on any modules. Its primary goal is to enrich the user experience by offering a clear visual representation of code progression. By showcasing a visual cue of completion, it simplifies the monitoring of code execution, particularly for lengthy processes. This feature empowers users to effortlessly track their code's advancement, thus fostering smoother workflow management and ensuring timely task completion. The module's lack of dependency on external libraries enhances its accessibility and usability, rendering it an invaluable asset for developers aiming to optimize their coding workflow and bolster productivity.

## Parameters
**status** (optional): If set to **True**, displays a completion message when the iteration is finished.

**symbol** (optional): The symbol used to represent progress in the bar. Default is "#".

**color** (optional): If set to **True**, the progress bar will be colored. Default is **False**.

For more details please check the code below!

## Full Code
Here is the full progress bar module code:
```python
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
```

## Example Use Code
Here is the example code to show how you can use it:
```python
from CustomProgressBar import custom_progress_bar as bar

for i in bar(range(1, 10000), status=True, symbol="#", color=True):
    pass
```

## How To Report Problems
If you encounter any errors or bugs in the code, please create an issue.

## License
This project is under the [MIT License](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) - see the [LICENSE.md](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) file for details.
