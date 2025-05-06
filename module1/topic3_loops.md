# Loops in Python

## Why Loops?
Repeat actions—ideal for scanning multiple stocks or price points.

## `for` Loop
```python
for item in iterable:
    # do something with item
```

## `while` Loop
```python
while condition:
    # repeat until condition is False
```

## Control Statements
- `break` → exit loop
- `continue` → skip to next iteration

## Examples

1. **Print Prices**
   ```python
   prices = [3299.75, 3500.50, 2980.00]
   for p in prices:
       print("Price:", p)
   ```
2. **Scan Symbols**
   ```python
   symbols = ["TCS","RELIANCE","INFY"]
   for s in symbols:
       print("Checking", s)
   ```
3. **While-Countdown**
   ```python
   count = 5
   while count > 0:
       print("Next fetch in", count, "sec")
       count -= 1
   print("Fetching data now!")
   ```

> **Visual Placeholder:** loop flow diagram.
