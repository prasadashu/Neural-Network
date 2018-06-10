import numpy as np

alphas = [0.001,0.01,0.1,1,10,100,1000]

# compute sigmoid nonlinearity
def sigmoid(x, deriv = False):
    if(deriv == True):
        return x * (1 - x)
    output = 1/(1+np.exp(-x))
    return output

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],
			[1],
			[1],
			[0]])

for alpha in alphas:
    print ("\nTraining With Alpha:" + str(alpha))
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((3,4)) - 1
    synapse_1 = 2*np.random.random((4,1)) - 1
    temp = 0

    for j in range(60000):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2
        temp = layer_2_error ** 2

        if (j% 10000) == 0:
           print ("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))))

        layer_2_delta = layer_2_error*sigmoid(layer_2, True)
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        layer_1_delta = layer_1_error * sigmoid(layer_1, True)

        synapse_1 += alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 += alpha * (layer_0.T.dot(layer_1_delta))
    
    print("Cost of the process: ", np.mean(temp))