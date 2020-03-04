import numpy as np
import pickle, random

def load(model_file):
    """
    Loads the network from the model_file
    :param model_file: file onto which the network is saved
    :return: the network
    """
    return pickle.load(open(model_file))

class NeuralNetwork(object):
    """
    Implementation of an Artificial Neural Network
    """
    def __init__(self, input_dim, hidden1_size, hidden2_size, output_dim, learning_rate=0.01):
        """
        Initialize the network with input, output sizes, weights and biases
        :param input_dim: input dim
        :param hidden_size: number of hidden units
        :param output_dim: output dim
        :param learning_rate: learning rate alpha
        :return: None
        """
        #Rohit debug comments : assume input = 16, hidden1 = 8, hidden2 = 4, output = 2
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        #Rohit debug comments : shape (8,16)
        self.Wxh1 = np.random.randn(self.hidden1_size, self.input_dim) * 0.01 # Weight matrix for input layer to first hidden layer
        #Rohit debug comments : shape: (4,8)
        self.Wh1h2 = np.random.randn(self.hidden2_size,self.hidden1_size) * 0.01 # Weight matric for input to second hidden layer
        #Rohit debug comments : shape: (2,4)
        self.Wh2y = np.random.randn(self.output_dim, self.hidden2_size) * 0.01 # Weight matrix for second hidden layer to output layer
        #Rohit debug comments : shape: (8,1)
        self.bh1 = np.zeros((self.hidden1_size, 1)) # first hidden layer bias
        #Rohit debug comments : shape: (4,1)
        self.bh2 = np.zeros((self.hidden2_size, 1)) # second hidden layer bias
        #Rohit debug comments : shape: (2,1)
        self.by = np.zeros((self.output_dim, 1)) # output bias
        self.learning_rate = learning_rate

    def _feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        # pdb.set_trace()
        #Rohit debug comments : shape: (8,1)
        h1_a = np.tanh(np.dot(self.Wxh1,np.reshape(X,(len(X),1))) + self.bh1)
        #Rohit debug comments : shape: (4,1)
        h2_a = np.tanh(np.dot(self.Wh1h2, h1_a) + self.bh2)
        #Rohit debug comments : shape: (2,1)
        ys = np.exp(np.dot(self.Wh2y, h2_a) + self.by)
        #Rohit debug comments : shape: (2,1)
        probs = ys/np.sum(ys)
        return h1_a, h2_a, probs

    def _update_parameter(self, dWxh1, dbh1, dWh1h2, dbh2, dWh2y, dby):
        """
        Update the weights and biases during gradient descent
        :param dWxh: weight derivative from input to hidden
        :param dbh: bias derivative from input to hidden
        :param dWhy: weight derivative from hidden to output
        :param dby: bias derivative from hidden to output
        :return: None
        """
        self.Wxh1 += -self.learning_rate * dWxh1
        self.bh1 += -self.learning_rate * dbh1
        self.Wh1h2 += -self.learning_rate * dWh1h2
        self.bh2 += -self.learning_rate * dbh2
        self.Wh2y += -self.learning_rate * dWh2y
        self.by += -self.learning_rate * dby

    def _back_propagation(self, X, t, h1_a, h2_a, probs):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        dWxh1, dWh1h2, dWh2y = np.zeros_like(self.Wxh1), np.zeros_like(self.Wh1h2), np.zeros_like(self.Wh2y)
        dbh1, dbh2, dby = np.zeros_like(self.bh1), np.zeros_like(self.bh2), np.zeros_like(self.by)

        #error calculation in the last layer
        dy = np.copy(probs)
        dy[t] -= 1
        #Rohit debug comments : shape: (2,4)
        dWh2y = np.dot(dy, h2_a.T)
        dby += dy

        #error calculation in second hidden layer
        #Rohit debug comments : shape: (4,1)
        dh2 = np.dot(self.Wh2y.T,dy)
        dh2raw = (1 - h2_a * h2_a) * dh2 
        dbh2 += dh2raw
        #Rohit debug comments : shape: (4,8)
        dWh1h2 += np.dot(dh2raw, h1_a.T)

        #error calculation in the first hidden layer
        #Rohit debug comments : shape: (8,1)
        dh1 = np.dot(self.Wh1h2.T, dh2raw)  # backprop into h
        dh1raw = (1 - h1_a * h1_a) * dh1 # backprop through tanh nonlinearity
        dbh1 += dh1raw
        dWxh1 += np.dot(dh1raw, np.reshape(X, (len(X), 1)).T)
        return dWxh1, dWh1h2, dWh2y, dbh1, dbh2, dby

    def _calc_smooth_loss(self, loss, len_examples):
        """
        Calculate the smoothened loss over the set of examples
        :param loss: loss calculated for a sample
        :param len_examples: total number of samples in training + validation set
        :return: smooth loss
        """

        return 1./len_examples * loss

    def train(self, inputs, targets, num_epochs):
        """
        Trains the network by performing forward pass followed by backpropagation
        :param inputs: list of training inputs
        :param targets: list of corresponding training targets
        :param validation_data: tuple of (X,y) where X and y are inputs and targets
        :param num_epochs: number of epochs for training the model
        :return: None
        """
        for k in range(num_epochs):
            loss = 0
            for i in range(len(inputs)):
                # Forward pass
                h1_a, h2_a, probs = self._feed_forward(inputs[i])
                loss += -np.log(probs[targets[i], 0])

                # Backpropagation
                dWxh1, dWh1h2, dWh2y, dbh1, dbh2, dby = self._back_propagation(inputs[i], targets[i], h1_a, h2_a, probs)

                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh1, dbh1, dWh1h2, dbh2, dWh2y, dby)

            # validation using the validation data

            '''validation_inputs = validation_data[0]
            validation_targets = validation_data[1]

            for i in range(len(validation_inputs)):
                # Forward pass
                h1_a, h2_a, probs = self._feed_forward(validation_inputs[i])
                loss += -np.log(probs[validation_targets[i], 0])

                # Backpropagation
                dWxh1, dWh1h2, dWh2y, dbh1, dbh2, dby = self._back_propagation(validation_inputs[i], validation_targets[i], h1_a, h2_a, probs)

                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh1, dbh1, dWh1h2, dbh2, dWh2y, dby)'''

            if k%1 == 0:
                print("Epoch " + str(k) + " : Loss = " + str(self._calc_smooth_loss(loss, len(inputs))))


    def predict(self, X):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        h1_a, h2_a, probs = self._feed_forward(X)
        # return probs
        return np.argmax(probs)

    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

if __name__ == "__main__":
    nn = NeuralNetwork(8,4,2,8)
    inputs = []
    targets = []
    for i in range(1000):
        num = random.randint(0,7)
        inp = np.zeros((8,))
        inp[num] = 1
        inputs.append(inp)
        targets.append(num)

    nn.train(inputs[:800], targets[:800], (inputs[800:], targets[800:]), 50)
    print(nn.predict([0,1,0,0,0,0,0,0]))
    print(nn.predict([0,0,0,1,0,0,0,0]))
    print(nn.predict([0,0,0,0,0,1,0,0]))
    print(nn.predict([0,0,0,0,0,0,0,1]))