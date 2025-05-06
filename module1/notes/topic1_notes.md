# Introduction to Python & APIs

## 1.1 Installing & Running Python

- **Windows:**
  1. Go to https://python.org/downloads/windows → Download the latest **Windows installer**
  2. Run it and **check “Add Python to PATH”**
  3. Open Command Prompt (Win + R → `cmd`)
  4. Type  
     ```
     python --version
     ```
     You should see something like `Python 3.10.x`.

- **Mac:**
  1. Open Terminal
  2. Install via Homebrew:
     ```
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     brew install python
     ```
  3. Verify:  
     ```
     python3 --version
     ```

- **Linux (Ubuntu/Debian):**
  ```
  sudo apt update
  sudo apt install python3 python3-pip
  python3 --version
  ```

## 1.2 Writing Your First Program

1. Open your code editor (e.g. VS Code) or simply your terminal.
2. Create a file named `hello.py` containing:
   ```python
   print("Hello, World!")
   ```
3. Run it:
   ```
   python hello.py
   ```
   ✅ You should see:
   ```
   Hello, World!
   ```

## 1.3 Basic Concepts

| Concept      | What It Means                        | Example                       |
|--------------|--------------------------------------|-------------------------------|
| **Variable** | A name for storing data              | `name = "Aman"`               |
| **Data Types** | Numbers, text, true/false          | `42`, `"Archit"`, `True`      |
| **Operator** | Symbols that do math or combine data | `+`, `-`, `*`, `/`, `==`      |
| **Function** | A reusable block of code             | `def greet(): print("Hi")`    |

## 1.4 Simple Examples

1. **Variables & Arithmetic**
   ```python
   a = 10
   b = 3
   print("Sum:", a + b)        # Sum: 13
   print("Divide:", a / b)     # Divide: 3.333...
   ```

2. **Strings & Concatenation**
   ```python
   first = "Upstox"
   second = "API"
   print(first + " " + second) # Upstox API
   ```

3. **User Input**
   ```python
   name = input("Enter your name: ")
   print("Welcome,", name)
   ```

## 1.5 What Is an API?

**Definition**  
An **Application Programming Interface** is a contract between two software components.

**Analogy**  
Like a waiter in a restaurant: you place an order (request), the kitchen prepares it, the waiter brings back your dish (response).
