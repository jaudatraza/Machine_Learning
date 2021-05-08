class LogisticRegression:
    
    def fit(self, X, y, lr = 0.001, epochs=10000, verbose=True, batch_size=1):
        self.classes = np.unique(y)
        # get classes unique value
        y = (y==self.classes[1]) * 1
        X = self.add_bias(X)
        # Init weights to zero
        self.weights = np.zeros(X.shape[1])
        self.loss = []
        for i in range(epochs):
            #gets linear combination of input features and weights, apply sigmoid function
            self.loss.append(self.cross_entropy(X,y))
            if i % 1000 == 0 and verbose: 
                print('Iterations: %d - Error : %.4f' %(i, self.loss[i]))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch =  X[idx], y[idx]
            #Compute the gradient of the cost function with respect to the weight vector and bias.
            #Update the weights and bias
            self.weights -= lr * self.get_gradient(X_batch, y_batch)
        return self
    
    def get_gradient(self, X, y):
        '''
        Compute the gradient of the cost function with respect to the weight vector
        '''
        return -1.0 * (y - self.predict_(X)).dot(X) / len(X)
    
    def predict_(self, X):
        '''
        Take the data matrix and do the dot product with the wieight and send with the activation function
        '''
        return self.sigmoid(np.dot(X, self.weights))
    
    def predict(self, X):
        '''
        Return Prediction
        '''
        return self.predict_(self.add_bias(X))
    
    def sigmoid(self, z):
        ''' 
        logistic(sigmoid) function for activation function
        '''
        return 1.0/(1 + np.exp(-z))
    
    def predict_classes(self, X):
        '''
        Return the classification
        '''
        return self.predict_classes_(self.add_bias(X))

    def predict_classes_(self, X):
        '''
        Depending on close to which probablibily prediction falls on select the class
        '''
        return np.vectorize(lambda c: self.classes[1] if c>=0.5 else self.classes[0])(self.predict_(X))
    
    def cross_entropy(self, X, y):
        '''
        the cross-entropy loss function
        '''
        p = self.predict_(X)
        return (-1 / len(y)) * (y * np.log(p)).sum()

    def add_bias(self,X):
        '''
        This is to add bias to the Data
        '''
        return np.insert(X, 0, 1, axis=1)

    def score(self, X, y):
        '''
        TReturns the accracy score of algo using the loss fucntion. 
        '''
        return self.cross_entropy(self.add_bias(X), y)
        
from sklearn.metrics import accuracy_score
def train_model(X, y, model):
'''
Get the model to run with the data and print the accuracy of the prediction compare to previous. 
'''
    model.fit(X, y, lr=0.1)
    pre = model.predict_classes(X)
    print('Accuracy :: ', accuracy_score(y, pre))
    
def LogisticRegressionRegular(Dataset):
	'''
	This is the main function that split the data and the put it into the algo and create a graph
	'''
    X = Dataset.iloc[:, :-1]
    Y = Dataset.iloc[:, -1]
    X = X.to_numpy()
    Y = Y.to_numpy()
    lr = LogisticRegression()
    train_model(X,(Y !=0 )*1, lr)
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(lr.loss)), lr.loss)
    plt.title("Development of cost over training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()
    
LogisticRegressionRegular(Iris)   
    
    
    
    
    
    
    
    
    
    
    
