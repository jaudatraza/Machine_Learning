import numpy as np
import matplotlib.pyplot as plt

class MultiClassLogisticRegression:
    
    def __init__(self, n_iter = 10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres
    
    def fit(self, X, y, batch_size=64, lr=0.001, rand_seed=4, verbose=False): 
        np.random.seed(rand_seed) 
        # Get all the classes in the Data
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        # Add Bias to Data Set
        X = self.add_bias(X)
        # One hot encode the class
        y = self.one_hot(y)
        self.loss = []
        #initialize the Weight matrix to zero
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
        self.fit_data(X, y, batch_size, lr, verbose)
        return self
 
    def fit_data(self, X, y, batch_size, lr, verbose):
        i = 0
        while (not self.n_iter or i < self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.predict_(X_batch)
            update = (lr * np.dot(error.T, X_batch))
            self.weights += update
            if np.abs(update).max() < self.thres: break
            if i % 1000 == 0 and verbose: 
                print(' Training Accuray at {} iterations is {}'.format(i, self.evaluate_(X, y)))
            i +=1
    
    def predict(self, X):
        '''
        Return Preidction
        '''
        return self.predict_(self.add_bias(X))
    
    def predict_(self, X):
        '''
        Take the data matrix and do the dot product with the wieight and send with the activation function
        '''
        pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)
    
    def softmax(self, z):
        '''
        Softmax function as an activation function
        '''
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

    def predict_classes(self, X):
        '''
        Depending on close to which probablibily prediction falls on select the class
        '''
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
  
    def add_bias(self,X):
        '''
        This is to add bias to the Data
        '''
        return np.insert(X, 0, 1, axis=1)

    def score(self, X, y):
        '''
        Get the accuracy_score of the algo
        '''
        return np.mean(self.predict_classes(X) == y)
    
    def one_hot(self, y):
        '''
        In the formula to compute the cost over the training, which is done by cross entropy and in that the target label is one hot encoded.
        '''
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
    
    def evaluate_(self, X, y):
        '''
        Evaludte the performace of the algo
        '''
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))
    
    def cross_entropy(self, y, probs):
        '''
        the cross-entropy loss function
        '''
        return -1 * np.mean(y * np.log(probs))
        
        
def MulticlassLogReg(Data, feature1, feature2): 
'''
This is the main function that needs to take the data and run the classification with 2 features
This prints the result in color mesh which also show which class is which color and were they are in teh map
'''   
    X = Data[[feature1, feature2]].copy()
    Y = Data.iloc[:, -1]
    X = X.to_numpy()
    Y = Y.to_numpy()
    logreg = MultiClassLogisticRegression()

    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.1  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict_classes(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    c = plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)
    plt.colorbar(c) 

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    #Set the X and Y limit of the plot
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()
    print(logreg.score(X, Y))
    # Plots loss each iteration 
    fig = plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(logreg.loss)), logreg.loss)
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()        
        
        
# How the function is called 
# Takes in Data Frame       
MulticlassLogReg(Iris,'Sepal_L', 'Sepal_W')        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
