# Conditional Programming

## What Is Conditional Programming?
You tell your program to decide between two or more paths.

- **Syntax**
  ```python
  if condition:
      # run this block
  elif other_condition:
      # run this block
  else:
      # run this block
  ```
- **Key Points**
  - Every `if` must end with a colon (`:`)
  - Indent by 4 spaces
  - `elif` and `else` are optional

## Examples (Indian Stocks)

1. **Single Decision**
   ```python
   price = 3500  # ₹3,500
   threshold = 3600
   if price < threshold:
       print("Consider BUYING TCS at ₹", price)
   else:
       print("Hold – price is ₹", price)
   ```

2. **Multiple Conditions**
   ```python
   price = 2800
   if price < 2500:
       print("Strong BUY for RELIANCE")
   elif price < 3000:
       print("Watch RELIANCE closely")
   else:
       print("Consider SELLING RELIANCE")
   ```

3. **Nested Conditions**
   ```python
   price = 1000
   market_open = True
   if market_open:
       if price < 1100:
           print("Buy INFY at dip")
       else:
           print("Wait for correction")
   else:
       print("Market closed")
   ```

> **Visual Placeholder:** flowchart showing `if → elif → else`.
