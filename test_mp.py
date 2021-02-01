import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score,classification_report
np.random.seed(42)
#getting data ready
heart_disease=pd.read_csv("heart-disease.csv")
x=heart_disease.drop("target",1)
y=heart_disease["target"]
X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.2)
core=heart_disease.corr()


#models
models={"RandomForest":RandomForestClassifier(),"Linear":LinearRegression(),"Neighbour":KNeighborsClassifier()}
#
base={}
for mod,i in models.items():
    i.fit(X_train,Y_train)
    base[mod]=i.score(x_test,y_test)
df=pd.DataFrame(base,index=[1])
df.plot(kind="bar")
fig,ax=plt.subplots(ncols=2,nrows=2)
ax[1,0]=sns.heatmap(core,annot=True,cmap= "winter")
scatter=ax[0,0].scatter(heart_disease["age"],heart_disease["target"],c=heart_disease["target"],cmap="winter")
ax[0,0].legend(*scatter.legend_elements(),title="Target")


plt.show()
#tuning





