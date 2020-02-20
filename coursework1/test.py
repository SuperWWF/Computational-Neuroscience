import numpy as np
a = np.array([[1,2,3,4]])
b = np.array([[1,2,3,4]])
c = [[0,3,4,5],[2,0,4,5],[2,3,4,5],[2,3,4,5]]
#print(b[0][:])
#print(np.dot(b[0][:],a))
print(a)
print(a.dot(a.T))
print(b.dot(b.T))
print(c)
print(b.dot(c).dot(b.T))
# for i in range(8):
#     c+=2
#     print(c)
