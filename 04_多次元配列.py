# 多次元配列

## 1次元配列
import numpy as np
print("1次元配列")
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

## 2次元配列
print("2次元配列")
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)

## 行列の積
print("行列の積")
A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)
print(np.dot(A, B))

C = np.array([[1, 2], [3, 4]])
print(C.shape)
print(A.shape)
# print(np.dot(A, C)) # ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)

A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)
B = np.array([7, 8])
print(B.shape)
print(np.dot(A, B))

# ニューラルネットワークの行列の積
print("ニューラルネットワークの行列の積")
X = np.array([1, 2])
print(X)
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)
Y = np.dot(X, W)
print(Y)
