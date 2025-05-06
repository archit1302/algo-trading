# Solutions for Topic 7

# 1.
def safe_int_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")

# 2.
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Cannot divide by zero.")
        return None

# 3.
def dict_lookup(d, key):
    try:
        return d[key]
    except KeyError:
        print(f"Missing key: {key}")
        return None

# 4. Combined
if __name__=="__main__":
    a = safe_int_input("Enter numerator: ")
    b = safe_int_input("Enter denominator: ")
    res = safe_divide(a, b)
    print("Division result:", res)
    sample = {"10": "Ten", "20": "Twenty"}
    print("Lookup:", dict_lookup(sample, str(res)))
