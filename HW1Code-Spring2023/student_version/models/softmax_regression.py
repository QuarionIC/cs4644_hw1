# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => ReLU activation => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        lay0 = np.zeros((X.shape[0], self.num_classes)).tolist()
        tposed = np.transpose(self.weights['W1'])
        for i in range(X.shape[0]):
            for j in range(self.num_classes):
                # print("Here!")
                # print("i:", i)
                # print("j:", j)
                lay0[i][j] += np.sum(X[i] * tposed[j])
        lay0 = np.array(lay0)
        lay1 = _baseNetwork.ReLU(self, lay0)
        lay2 = _baseNetwork.softmax(self, lay1)
        loss = _baseNetwork.cross_entropy_loss(self, lay2, y)
        accuracy = _baseNetwork.compute_accuracy(self, lay2, y)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        labels = np.empty(lay2.shape)
        for i in range(lay2.shape[0]):
            for j in range(lay2.shape[1]):
                if(y[i] == j):
                    labels[i,j] = 1
                else:
                    labels[i,j] = 0
        cross_gradient = (lay2 - labels) / X.shape[0]
        next_gradient = _baseNetwork.ReLU_dev(self, lay0)
        prod = cross_gradient * next_gradient
        gradient = np.zeros((X.shape[1], self.num_classes)).tolist()
        xtposed = np.transpose(X)
        prodtposed = np.transpose(prod)
        # print(xtposed.shape)
        # print(prodtposed.shape)
        for i in range(X.shape[0]):
            for j in range(prodtposed.shape[0]):
                gradient[i][j] += np.sum(xtposed[i] * prodtposed[j])
        gradient = np.array(gradient)
        self.gradients['W1'] = gradient
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


