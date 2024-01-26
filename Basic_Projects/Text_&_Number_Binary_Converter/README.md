# MY PRACTICE PROJECT 1 USING PROCEDURAL PROGRAMMING

This is a simple practice project showcasing procedural programming in Python. The project includes a function `convert_to_binary` that takes an input (alphabet or number) and converts it into its binary representation using the ASCII values of individual characters.

## Full code:

```python
# MY PRACTICE PROJECT 1 USING PROCEDURAL PROGRAMMING
def convert_to_binary(input):
    get_binary = ' '.join(format(ord(character), 'b') for character in input)
    return get_binary

user_input = input("Enter an alphabet or number: ")
binary_result = convert_to_binary(user_input)
print(f"Binary representation: {binary_result}")
```

If you find this project interesting, feel free to use and modify it for integration into more complex programs.
