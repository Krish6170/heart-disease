#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
from  sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,roc_curve,recall_score,confusion_matrix
from  sklearn.model_selection import cross_val_score
np.random.seed(42)
#loading data
heart_disease=pd.read_csv("heart-disease.csv")

#feATURE
#modeling
models={"kneighbor":KNeighborsClassifier(),"logistic":LogisticRegression(),"randomforest":RandomForestClassifier()}
def eval(model,xtest,ytest):
    bscore=model.score(x_test,y_test)
    return  bscore
basescores={}
#splitting
x=heart_disease.drop("target",axis=1)
y=heart_disease["target"]
x_train, x_test, y_train ,y_test=train_test_split(x,y,test_size=0.2)

for mod,i in models.items():
    i.fit(x_train,y_train)
    basescores[mod]=eval(i,x_test,y_test)

base=pd.DataFrame(basescores,index=[1])
base.plot(kind="bar")
print(base)
#tuning
randomforest={"n_estimators":[100,150,200,250,300,350,400,450,500],
              "max_features":["auto", "sqrt", "log2"],
              "max_depth":[1,2,3,4,5,6,7,8,9,10,11,12,13],
              "min_samples_leaf":[2,4,6],
              "min_impurity_decrease":[.1,.2,.3,.4,.5,.6,.7,.8,.9],
              }
logisticregression={"solver": ["newton-cg", "lbfgs","liblinear"],
                    "penalty":["l1", "l2", "elasticnet"],
                     "C":[1,4.15108681461894e+81, 0.23357214690901212]}
gd_rf=RandomizedSearchCV(estimator=RandomForestClassifier(),
                         param_distributions=randomforest,
                         cv=5,
                         n_iter=10)
gd_lr=RandomizedSearchCV(estimator=LogisticRegression(),
                         param_distributions=logisticregression,
                         cv=5,
                         n_iter=20)

gd_rf.fit(x_train,y_train)
gd_lr.fit(x_train,y_train)


print(gd_rf.best_params_,gd_rf.score(x_test,y_test))
print(gd_lr.best_params_,gd_lr.score(x_test,y_test))
import pickle
filename1=r"C:\coding_stuff\milestonepro_py\randomf.pkl"
filename2=r"C:\coding_stuff\milestonepro_py\logistics.pkl"
pickle.dump(gd_rf,open(filename1,'wb'))
pickle.dump(gd_lr,open(filename2,'wb'))
y_pred=gd_rf.predict(x_test)
print(classification_report(y_test,y_pred))