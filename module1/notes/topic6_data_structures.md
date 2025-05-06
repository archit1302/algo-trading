# Lists & Dictionaries

## Lists
- Ordered, mutable  
- Created with `[ ]`  
- Example: `prices = [3299.75, 3300.00]`

## Dictionaries
- Key-value pairs  
- Created with `{ }`  
- Example: `stock = {"symbol":"TCS","price":3299.75}`

## Nested Structures
- List of dicts or dict of lists  
- Example:
  ```python
  market = [
      {"symbol":"TCS","price":3299.75},
      {"symbol":"INFY","price":1200.00},
  ]
  ```

## Operations
- **Add to list:** `prices.append(3310)`  
- **Access dict:** `stock["price"]`  
- **Iterate:**
  ```python
  for s in market:
      print(s["symbol"], s["price"])
  ```

> **Visual Placeholder:** Venn of lists vs dicts.
