# Solutions for Topic 1 Assignments

# hello.py
print("Hello, Aman! Welcome to Python Bootcamp.")

# calc.py
a = float(input("Enter first number: "))
b = float(input("Enter second number: "))
print("Sum:", a + b)
print("Difference:", a - b)
print("Product:", a * b)
print("Quotient:", a / b if b != 0 else "Cannot divide by zero")

# join_names.py
first = input("First name: ")
last = input("Last name: ")
print(f"Hello, {first} {last}!")

# utils.py
def square(x):
    """Return the square of x."""
    return x * x

# test_square.py
from utils import square
num = float(input("Enter a number to square: "))
print(f"{num}² = {square(num)}")
