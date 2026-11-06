import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

print("Original array:", arr)

print("\narr[2:5]:", arr[2:5])

print("arr[:4]:", arr[:4])

# Slicing from index 5 to end
print("arr[5:]:", arr[5:])

# Slicing with step - every 2nd element
print("arr[::2]:", arr[::2])

# Slicing with negative indices
print("arr[-3:]:", arr[-3:])

# Slicing with step backward
print("arr[::-1]:", arr[::-1])

# 2D array slicing
arr2d = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print("\n2D Array:\n", arr2d)

# Get first row
print("arr2d[0]:", arr2d[0])

# Get element at row 1, column 2
print("arr2d[1, 2]:", arr2d[1, 2])

# Get all elements in column 1
print("arr2d[:, 1]:", arr2d[:, 1])

# Get first 2 rows and all columns
print("arr2d[:2, :]:\n", arr2d[:2, :])
