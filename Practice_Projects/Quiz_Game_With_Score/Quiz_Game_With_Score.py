def ask_question(question, correct_option):
    print(question)
    user_answer = input("Enter your answer: ").lower()
    return user_answer == correct_option

def run_quiz(questions):
    score = 0
    for i, (question, correct_option) in enumerate(questions, start=1):
        if ask_question(question, correct_option):
            score += 1
    return score

questions = [
    ("Q.1. What is the purpose of the '__init__' method in Python classes?\na) To initialize class attributes b) To define the class constructor\nc) To create a new instance of the class d) To destroy the class object", "b"),

    ("Q.2. How can you open a file named 'example.txt' in Python for reading?\na) open('example.txt', 'r') b) read('example.txt')\nc) file('example.txt', 'read') d) open_file('example.txt', 'read')", "a"),

    ("Q.3. What does the 'self' keyword refer to in a Python class method?\na) It represents the class itself b) It represents the instance of the class\nc) It refers to a static method d) It is a reserved keyword", "b"),

    ("Q.4. Which of the following is true about Python's list comprehension?\na) It can only be used for lists b) It cannot contain an 'if' statement\nc) It always results in a tuple d) It provides a concise way to create lists", "d"),

    ("Q.5. What does the 'pip' tool in Python stand for?\na) Python Install Package b) Package Installation Python\nc) Pip Installs Python d) Python Package Installer", "a"),

    ("Q.6. How can you check the length of a list named 'my_list' in Python?\na) length(my_list) b) len(my_list)\nc) count(my_list) d) size(my_list)", "b"),

    ("Q.7. What is the purpose of the 'try' and 'except' blocks in Python?\na) To define a loop b) To handle exceptions and errors\nc) To create a new function d) To import external modules", "b"),

    ("Q.8. In Python, what is the difference between '==' and 'is' for comparing objects?\na) They are the same b) '==' checks for equality, 'is' checks for identity\nc) 'is' checks for equality, '==' checks for identity d) They both check for identity", "b"),

    ("Q.9. How can you remove an item with a specific value from a Python list?\na) list.remove(value) b) list.delete(value)\nc) list.pop(value) d) list.exclude(value)", "a"),

    ("Q.10. What is the purpose of the '__str__' method in Python?\na) To convert an object to a string b) To define a string variable\nc) To format a string d) To remove trailing whitespaces", "a"),
]

user_score = run_quiz(questions)
print(f"Your score: {user_score}")
