# Main Project 5

## Overview
This code implements a flexible and customizable progress bar generator in Python. The progress bar is designed to visually represent the completion status of an iterable process. Users can control various aspects of the progress bar, such as symbol, color, and status messages.

## Functionality
The progress bar generator includes the following parameters:

- `iterable`: The iterable process for which the progress bar is generated.
- `status` (optional, default `False`): Indicates whether to display a completion message.
- `symbol` (optional, default `#`): The symbol used to represent progress in the bar.
- `color` (optional, default `False`): Enables color-coding for a visually enhanced display.

## Usage
```python
from Custom_Progress_Bar import Custom_Progress_Bar

# Example Usage
for item in Custom_Progress_Bar(iterable, status=True, symbol="#", color=True):
    # Your iterable process here
    pass
```

## Parameters
- `iterable`: The iterable process to track progress.
- `status`: Set to `True` to display a completion message. (Optional, default `False`)
- `symbol`: The symbol used for progress in the bar. (Optional, default `#`)
- `color`: Enable color-coding for a visually enhanced display. (Optional, default `False`)

## Example
```python
from Custom_Progress_Bar import Custom_Progress_Bar

# Example Usage
data = range(100)
for item in Custom_Progress_Bar(data, status=True, symbol="#", color=True):
    # Simulate a time-consuming process
    # Your code here
```

## Full Code:

```python
def Custom_Progress_Bar(iterable, status=False, symbol="#", color=False):
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
	        
