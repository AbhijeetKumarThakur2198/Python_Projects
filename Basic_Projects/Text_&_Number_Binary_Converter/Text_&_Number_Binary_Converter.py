# MY PRACTICE PROJECT 1 USING PROCEDURAL PROGRAMMING
def convert_to_binary(input):
    get_binary = ' '.join(format(ord(character), 'b') for character in input)
    return get_binary

user_input = input("Enter an alphabet or number: ")
binary_result = convert_to_binary(user_input)
print(f"Binary representation: {binary_result}")
