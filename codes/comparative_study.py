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
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

#==============================================================================
def k_fold(model, x, y_true):
    model.fit(x, y_true)
    y_pred = model.predict(x)
    a = accuracy_score(y_true, y_pred, normalize=False)
    b = 500 - a
    return [precision_recall_fscore_support(np.array(y), np.array(y_pred), average='weighted'),
            a, b]
      
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

   
    print('Logistic Regression precision recall f measure correct incorrect:', val[0])
    print('knn precision recall f measure correct incorrect:', val[1])
    print('Gaussian Naive Bayes precision recall f measure correct incorrect:', val[2])
    print('Multinominal Naive Bayes precision recall f measure correct incorrect:', val[3])
    print('Bernoulli Naive Bayes precision recall f measure correct incorrect:', val[4])
    print('Decision Tree(CART) precision recall f measure correct incorrect:', val[5])
    print('Decision Tree(ID3) precision recall f measure correct incorrect:', val[6])
    print('Random Forest precision recall f measure correct incorrect:', val[7])
    print('Support Vector Machine precision recall f measure correct incorrect:', val[8])
    print('MLP Neural Network precision recall f measure correct incorrect:', val[9])
    
    