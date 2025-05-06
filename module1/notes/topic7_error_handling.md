# Error Handling

## Why?
User input & calculations can fail—avoid crashes.

## Syntax
```python
try:
    # risky code
except SomeError:
    # handle it
else:
    # runs if no error
finally:
    # always runs
```

## Common Exceptions
- `ValueError`: bad conversion  
- `ZeroDivisionError`: divide by zero  
- `KeyError`: dict missing key

## Examples

1. **Input Validation**
   ```python
   try:
       x = int(input("Enter number: "))
   except ValueError:
       print("Not a valid number")
   ```
2. **Safe Division**
   ```python
   try:
       result = a / b
   except ZeroDivisionError:
       result = None
       print("Cannot divide by zero")
   ```

> **Visual Placeholder:** exception flowchart.
