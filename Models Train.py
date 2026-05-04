from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
def get_models():
  models = dict()
  models['dt_ent'] = DecisionTreeClassifier(criterion='entropy')
  models['dt_gini'] = DecisionTreeClassifier(criterion='gini')
  models['mlr']=LogisticRegression()
  models['lsvc']=SVC(kernel='linear')
  models['rsvc']=SVC()
  models['ssvc']=SVC(kernel='sigmoid')
  models['psvc']=SVC(kernel='poly')
  models['rf'] = RandomForestClassifier()
  return models
  models=get_models()
  models.items()


import numpy as np
def evaluate_model(model, X, y,m,s):
  acc = [] ### Blank vector
  for i in range(m):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=s, random_state=i)# split dataset
    m = model.fit(X_train,y_train) # fit the model
    YPred = m.predict(X_test) # predict
    a = accuracy_score(y_test,YPred) # compute accuracy
    acc.append(a) # append accuracy
    return np.mean(acc)

from numpy import mean, std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X_final, y, 10,0.2)
	results.append(scores)
	names.append(name)
	print(  (name, mean(scores)))