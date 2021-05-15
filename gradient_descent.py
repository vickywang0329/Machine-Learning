"""ADAptive LInear NEuron classifier
    Parameters
    ----------
    eta=float
     Learning rate (between 0.0 and 1.0)
    n_iter:int
     Passes over the training dataset
    random_state: int
     Random number generator seed for random weight
     initialization.

     Attributes
     ----------
     w_= 1d-array
       Weights after fitting
    cost_: list
       Sum of square cost function value in each epoch
"""
import numpy as np 
class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state= random_state

    def fit(self, X, y):
        rgen= np.random.RandomState(self.random_state)
        self.w_= rgen.normal(loc=0.0, scale=0.01, size=1+ X.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            net_input= self.net_input(X)
            output= self.activation(net_input)
            errors= (y-output)
            self.w_[1:]+= self.eta * X.T.dot(errors)
            self.w_[0] +=self.eta* errors.sum()
            cost= (errors*2).sum()/ 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """calculate net input"""
        return np.dot(X, self.w_[1:])+ self.w_[0]

    def activation(self, X):
        """compute linear activation"""
        return X
        
    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>=0.0,1,-1)
#引入X, y
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data', header=None, encoding='utf-8')
print(df.tail())
y= df.iloc[0:100, 4].values
y= np.where(y=='Iris-setosa', -1,1)
X= df.iloc[0:100, [0,2]].values
'''
plt.scatter(X[:50,0], X[:50,1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.legend(loc='upper left')
plt.show()
'''

#為兩個不同的學習速率，繪製每輪的成本圖形
import matplotlib.pyplot as plt 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error')
ax[0].set_title('Adaline- learning rate 0.01')

ada2= AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1),ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline- learning rate 0.0001')
plt.show()

#標準化的特徵縮放法
X_std = np.copy(X)
X_std[:,0]= (X[:,0]-X[:,0].mean())/ X[:,0].std()
X_std[:,1]= (X[:,0]-X[:,1].mean())/ X[:,1].std()


from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers=('s', 'x', 'o', '^', 'v')
    colors= ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap= ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max= X[:,0].min() -1, X[:,0].max() +1
    x2_min, x2_max= X[:,1].min() -1, X[:,1].max() +1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z= z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.3, cmap=cmap )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, marker=markers[idx], label=cl, edgecolor='black')

ada = AdalineGD(n_iter=15, eta=0.0001)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Dscent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-sqaured-error')
plt.tight_layout()
plt.show()    

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta= eta
        self.n_iter= n_iter
        self.w_initialized= False
        self.shuffle= shuffle
        self.random_state= random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y= self._shuffle(X, y)
            cost= []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/ len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initailize_weights(X.shape[1])
        if y.ravel().shape[0]>1:
            for xi, target in zip(X,y):
                self._upgrade_weights(xi, target)
            else:
                self._update_weights(X,y)
            return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_= self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output= self.activation(self.net_input(xi))
        error = (target-output)
        self.w_[1:]+= self.eta* xi.dot(error)
        self.w_[0]+= self.eta* error
        cost = 0.5* error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:])+ self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X))>=0.0, 1, -1)

#標準化的特徵縮放法
X_std = np.copy(X)
X_std[:,0]= (X[:,0]-X[:,0].mean())/ X[:,0].std()
X_std[:,1]= (X[:,0]-X[:,1].mean())/ X[:,1].std()

ada=AdalineGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline-Stochastic Gradient Dscent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1,len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average cost')
plt.tight_layout()
plt.show()



   
