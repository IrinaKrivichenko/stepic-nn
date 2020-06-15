import numpy as np

class nn:
    def __init__(self, sizes, output=True):
        print('shape of the neural network:')
        print(sizes)
        print(f'input layer | number of neurons {sizes[0]} ')
        self.list_of_layers = []
        np.random.seed(10)
        for x, y, i in zip(sizes[:-1], sizes[1:], range(len(sizes)-1)):
            W = np.random.randn(y, x)
            print(W)
            self.list_of_layers.append( self.layer(i+1,'sigmoid', W))

    def forward_pass(self, input_matrix):
        data = input_matrix
        for l in self.list_of_layers:
            data = l.forward_pass(data)
        return data

    def back_propagation(self, input_matrix, answers):
        predictions = self.forward_pass(input_matrix)
        errors = (predictions - answers).T
        for l in reversed(self.list_of_layers):
            errors = l.get_prev_layer_errors(errors)
        return errors

    class layer:
        def __init__(self, number_of_layer, actfunc, weights):
            '''weights - ndarray of shape (n_{l+1}, n_l) weights to neurons of layer'''
            self.number_of_layer = number_of_layer
            print(f'layer № {number_of_layer} | number of neurons {weights.shape[0]} | actfunc = {actfunc}')
            self.actfunc = actfunc
            self.weights = weights

        def diff_activation(self):
            if self.actfunc == 'sigmoid':
                sigmoid = 1 / (1 + np.exp(-self.sums))
                return sigmoid * (1 - sigmoid)
            if self.actfunc == 'relu':
                return np.where(self.sums < 0, 0, 1)

        def activation(self):
            if self.actfunc == 'sigmoid':
                return 1 / (1 + np.exp(-self.sums))
            if self.actfunc == 'relu':
                return np.max(self.sums, 0)

        def get_prev_layer_errors(self, error):
            """compute error on the previous layer of network
             error - ndarray of shape (n, n_{l+1})
                sums - ndarray of shape (n, n_l)
                weights - ndarray of shape (n_{l+1}, n_l)
            input - ndarray of shape (n, n_l) input of layer """

            delta_weights_for_node = (self.diff_activation() * error.T).T
            delta_weights = (delta_weights_for_node*self.input)
            print(f'delta_weights for layer {self.number_of_layer} :')
            print(delta_weights)
            print('nabla_w')
            err = self.weights.T.dot(delta_weights_for_node)
            self.weights = self.weights - delta_weights
            return err

        def forward_pass(self, input_matrix):
            #print(input_matrix.shape)
            #print(self.weights.shape)
            self.input = input_matrix
            self.sums = input_matrix.dot(self.weights.T)
            return self.activation()

n = nn([4,3,2])
x = np.array([[1, 2, 3, 4]])
y = np.array([[0, 1]])
n.back_propagation(x, y)

'''
my_nn = nn([3,2,1])
input = np.array([[3, 2, 1]])
answer = np.array([[1]])
for i in range(100):
    print(f' i= {i} prediction = {my_nn.forward_pass(input)}')
    my_nn.back_propagation(input, answer)
'''