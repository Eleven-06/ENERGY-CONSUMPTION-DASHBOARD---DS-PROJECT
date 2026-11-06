import numpy as np


def show_shapes():
	a1 = np.array([1, 2, 3, 4])
	a2 = np.array([[1, 2, 3], [4, 5, 6]])
	a3 = np.zeros((2, 3, 4))

	print("a1:", a1)
	print("a1.shape:", a1.shape)            # (4,)

	print("\na2:\n", a2)
	print("a2.shape:", a2.shape)            # (2, 3)
	print("rows (a2.shape[0]):", a2.shape[0])
	print("cols (a2.shape[1]):", a2.shape[1])

	print("\na3.shape:", a3.shape)            # (2, 3, 4)

	# Reshape example
	b = a1.reshape((2, 2))
	print("\nreshaped a1 to b (2,2):\n", b)
	print("b.shape:", b.shape)


if __name__ == "__main__":
	show_shapes()
