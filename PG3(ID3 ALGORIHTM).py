import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
iris=load_iris()
X=iris.data
Y=iris.target
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
clf=DecisionTreeClassifier(criterion="entropy",random_state=42)
clf.fit(X_train,Y_train)
tree_rules=export_text(clf,feature_names=iris.feature_names)
print("Decision tree:/n",tree_rules)
Y_pred=clf.predict(X_test)
accuracy=np.mean(Y_pred==Y_test)
print("accuracy:",accuracy)
