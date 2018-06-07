 class NeuralNetwork():

    def __init__(self, hidden_layer = 2):
        np.random.seed(1)
        self.hidden_layer = hidden_layer
        self.syn0 = 2 * np.random.random((4, self.hidden_layer)) - 1
        self.syn1 = 2 * np.random.random((self.hidden_layer, 1)) - 1

    def train(self, X_train, y_train):

        for j in range(60000):

            l0 = X_train
            l1 = sigmoid(np.dot(l0, self.syn0))
            l2 = sigmoid(np.dot(l1, self.syn1))

            #Backpropagation
            l2_error = y_train - l2
            #if(j % 10000) == 0:
                #print("Error: ", np.abs(l2_error))

            l2_delta = l2_error * sigmoid(l2, True)
            l1_error = np.dot(l2_delta, self.syn1.T)
            l1_delta = l1_error * sigmoid(l1, True)

            #Updating the synapses
            self.syn1 += np.dot(l1.T, l2_delta)
            self.syn0 += np.dot(l0.T, l1_delta)

    def test(self, X_test, y_test):
        l0 = X_test
        l1 = sigmoid(np.dot(l0, self.syn0))
        l2 = sigmoid(np.dot(l1, self.syn1))

        print("Error in test:")
        print(np.abs(y_test - l2))