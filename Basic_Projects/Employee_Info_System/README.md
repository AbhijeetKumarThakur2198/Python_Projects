# MY PRACTICE PROJECT 2 USING OBJECT ORIENTATION
This is a simple practice project illustrating the use of object-oriented programming in Python. The project defines a class `Employee` with attributes such as `first_name`, `middle_name`, `last_name`, `age`, `gender`, `currency`, `salary`, `address`, `email`, and `full_name`. The class also includes a method `report()` to generate a summary of the employee's details.

## FULL CODE:

```python
# MY PRACTICE PROJECT 2 USING OBJECT ORIENTATION
class Employee:
	def __init__(self, first_name="Unknown", middle_name="Unknown", last_name="Unknown", age=0, gender="", currency="₹", salary=0, address="Unknown"):
		self.first_name = first_name
		self.middle_name = middle_name
		self.last_name = last_name
		self.age = age
		self.gender = gender
		self.currency = currency
		self.salary = salary
		self.address = address
		self.email = first_name + middle_name + last_name + "4659" + "@example.mail.com"
		self.full_name = first_name + " " +  middle_name + " " + last_name

	def report(self):	       
           pronoun = "they are" 
           verb = "live" 
              
           if self.gender.lower() == "male":            
            pronoun = "he is"
            verb = "lives"
         
           elif self.gender.lower() == "female":           
            pronoun = "she is"   
            verb = "lives" 

           return f"{pronoun.capitalize()} {self.age} years old and {verb} at {self.address}. Total salary is {self.currency}{self.salary}."

try:								

	Employee1 = Employee("Abhijeet", "Kumar", "Thakur", 16, "Male", "₹", 50000, "C-10, Shiv Durga Vihar, Faridabad, Lakkarpur Haryana, 121009")

	Employee1_report = Employee1.report()

	print(Employee1.first_name)
	print(Employee1.middle_name)
	print(Employee1.last_name)
	print(Employee1.full_name)
	print(Employee1.age)
	print(Employee1.gender)
	print(Employee1.email)
	print(Employee1.address)
	print(Employee1.currency)
	print(Employee1.salary)
	print(Employee1_report)

except Exception as error:
	print(f"Error occured: {error}")
```

If you find this project interesting, feel free to use and modify it for integration into more complex programs.
