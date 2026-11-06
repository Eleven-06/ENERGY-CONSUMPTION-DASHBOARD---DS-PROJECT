import numpy as np
n1=np.array([1,2,3,4])
n2=np.array([5,6,7,8])
n3=np.concatenate((n1,n2))
print(np.array_split(n3, 4))