# Working with Strings

## Common Operations
- **Concatenate**: `"Hello" + " World"`
- **F-string**: `f"Price: ₹{price}"`
- **Split**: `"TCS,INFY,HDFC".split(",")`
- **Join**: `"-".join(["TCS","INFY"])`

## Examples

1. **Format Quote**
   ```python
   price = 3299.75
   print(f"TCS last traded at ₹{price:.2f}")
   ```
2. **Parse CSV Line**
   ```python
   line = "RELIANCE,2500,2550,2450,2525"
   symbol, o, h, l, c = line.split(",")
   ```
3. **Clean Input**
   ```python
   raw = "  INFY  "
   clean = raw.strip()
   ```

> **Visual Placeholder:** string methods mind-map.
