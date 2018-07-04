from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

#Normalization
X_train = (X_train - np.mean(X_train)) / (np.max(X_train) - np.min(X_train))
X_test = (X_test - np.mean(X_test)) / (np.max(X_test) - np.min(X_test))

y_test = y_test.reshape(y_test.shape[0], 1)
y_train = y_train.reshape(y_train.shape[0], 1)

np.random.seed(1)
syn0 = np.random.random((4, 3))
#bias_0 = np.zeros((1, 3))
syn1 = np.random.random((3, 1))
#bias_1 = np.zeros((1, 1))

def sigmoid(X, deriv = False):
	if(deriv == True):
		return X * (1 - X)
	return 1 / (1 + np.exp(-X))

def inverse_sinh(X, deriv = False):
	if(deriv == True):
		return 1 / (np.sqrt(1 + np.square(X)))
	return np.arcsinh(X)

for j in range(60000):
	l0 = X_train # 90x4 matrix
	l1 = inverse_sinh(np.dot(l0, syn0)) # 90x3 matrix
	l2 = inverse_sinh(np.dot(l1, syn1)) # 90x1 matrix

	learning_rate = 1

	#Backpropagation
	l2_error = y_train - l2 # 90x1 matrix
	l2_delta = l2_error * inverse_sinh(l2, True) #90x1 matrix
	l1_error = np.dot(l2_delta, syn1.T) # 90x3 matrix
	l1_delta = l1_error * inverse_sinh(l1, True) #90x3 matrix

	syn0 += learning_rate * (np.dot(l0.T, l1_delta))
	syn1 += learning_rate * (np.dot(l1.T, l2_delta))

#Changes are to be made here to improve the efficiency
def model_sigmoid(X):
	for i in range(len(X)):
		if X[i, 0] > 9.99999000e-01:
			X[i, 0] = 2
	for i in range(len(X)):
		if X[i, 0] < 9.99999999e-03:
			X[i, 0] = 0
	for i in range(len(X)):
		if X[i, 0] != 2 and X[i, 0] != 0:
			X[i, 0] = 1

def model_inverse_sinh(X):
	for i in range(len(X)):
		if X[i, 0] > 0.846:
			X[i, 0] = 1
	for i in range(len(X)):
		if X[i, 0] < 0.81:
			X[i, 0] = 0
	for i in range(len(X)):
		if X[i, 0] != 1 and X[i, 0] != 0:
			X[i, 0] = 2

#model_sigmoid(l2)
#model_inverse_sinh(l2)
#print(l2.T)
#print(y_train.T)

#Developing model for Training set
a = inverse_sinh(np.dot(X_train, syn0))

b = inverse_sinh(np.dot(a, syn1))

b = b * (1/10)

model_inverse_sinh(b)

#Training accuracy
print("Training set accuracy: ", 100 - (np.mean(np.abs(y_train - b)) * 100))

#Developing model for Testing set
a = inverse_sinh(np.dot(X_test, syn0))

b = inverse_sinh(np.dot(a, syn1))

b = b * (1/10)

model_inverse_sinh(b)

#Testing accuracy
print("Testing set accuracy: ", 100 - (np.mean(np.abs(y_test - b)) * 100))