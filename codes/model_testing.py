#=================== Imported packages & modules ==============================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from id3 import Id3Estimator
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as mt
import numpy as np
import matplotlib.pyplot as plt

#==============================================================================

def train_and_test(model, x, y):
    model.fit(x, y)
    res = model.predict(x)
    return mt.accuracy_score(res, y)
    
def train_tst_split(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 4)
    model.fit(x_train, y_train)
    res = model.predict(x_test)
    return mt.accuracy_score(res, y_test)
    
def k_fold(model, x, y):
    scr = cross_val_score(model, x, y, cv=10, scoring='accuracy')
    return scr.mean()

def ploting(val, objects):
    y_pos = np.arange(len(objects))
    plt.barh(y_pos, val, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Accuracy percentage')
    plt.ylabel('Models')
    plt.title('Model Accuracy')
    plt.show()
      
if __name__ == '__main__':
    file = 'first.xlsx'
    
    # Load spreadsheet
    data = pd.read_excel(file)
    
    #columns
    fcols = [
             'competitive programming background',
             'professional skill',
             'research background',
             'final year projct type',
             'enthusiasm',
             'teamwork ability',
             'communication & network skill',
             'cgpa'
    ]
    
    #selecting train data
    x = data[fcols]
    y = data['current job field']
    
    #initilizing the models
    lr = LogisticRegression()
    knn = KNeighborsClassifier(n_neighbors=5)
    gnb = GaussianNB()
    mn = MultinomialNB()
    ber = BernoulliNB()
    tree = DecisionTreeClassifier()
    id3 = Id3Estimator()
    rnd = RandomForestClassifier(n_estimators=300)
    svc = SVC()
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                        hidden_layer_sizes=(5, 75), random_state=1)
    
    """a = tree.fit(x,y)
    print(a.feature_importances_)"""
    val = list()
    val.append(k_fold(lr, x, y))
    val.append(k_fold(knn, x, y))
    val.append(k_fold(gnb, x, y))
    val.append(k_fold(mn, x, y))
    val.append(k_fold(ber, x, y))
    val.append(k_fold(tree, x, y))
    val.append(k_fold(id3, x, y))
    val.append(k_fold(rnd, x, y))
    val.append(k_fold(svc, x, y))
    val.append(k_fold(mlp, x, y))

    #k fold cross validation score
    print('k-fold')
    print('Logistic Regression accuracy score:', val[0])
    print('knn accuracy score:', val[1])
    print('Gaussian Naive Bayes accuracy score:', val[2])
    print('Multinominal Naive Bayes accuracy score:', val[3])
    print('Bernoulli Naive Bayes accuracy score:', val[4])
    print('Decision Tree(CART) accuracy score:', val[5])
    print('Decision Tree(ID3) accuracy score:', val[6])
    print('Random Forest accuracy score:', val[7])
    print('Support Vector Machine accuracy score:', val[8])
    print('MLP Neural Network accuracy score:', val[9])
    
    objects = ('LR','KNN','GNB','MNB','BNB','CART','ID3', 'RF','SVM', 'MLP')
    val2 = [i*100 for i in val]
    ploting(val2, objects)
    
    