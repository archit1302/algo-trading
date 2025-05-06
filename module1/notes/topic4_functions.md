# Functions

## Why Use Functions?
- Reuse code  
- Organize logic  
- Make your script cleaner

## Defining a Function
```python
def name(parameters):
    """docstring"""
    # code
    return result
```

## Calling a Function
```python
out = name(args)
```

## Examples

1. **Square**
   ```python
   def square(x):
       return x * x
   print(square(5))  # 25
   ```
2. **Return %**
   ```python
   def return_pct(buy, sell):
       return ((sell - buy) / buy) * 100
   print(return_pct(1000, 1100))  # 10.0
   ```
3. **Moving Average**
   ```python
   def sma(prices):
       return sum(prices) / len(prices)
   ```

> **Visual Placeholder:** call-stack diagram.
