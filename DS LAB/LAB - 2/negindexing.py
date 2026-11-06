import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print("Array:", arr)
print("Last element (index -1):", arr[-1])
print("Second last element (index -2):", arr[-2])
print("Third last element (index -3):", arr[-3])

arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print("\n2D Array:")
print(arr2d)
print("Last row (index -1):", arr2d[-1])
print("Last element of last row:", arr2d[-1, -1])
print("First element of last row:", arr2d[-1, 0])
print("Last column, first row:", arr2d[0, -1])

print("\nSlicing with negative indices:")
print("Last 2 elements:", arr[-2:])
print("All except last element:", arr[:-1])
print("From second last to second element:", arr[-4:-1])
