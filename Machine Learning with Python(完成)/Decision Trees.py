import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv("drug200.csv", delimiter=",")
# print(my_data[0:5])
# print(my_data.shape)

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# print(X[0:5])

from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

# print(X[0:5])

y = my_data["Drug"]
# print(y[0:5])
print(type(y))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
# print('Shape of X training set{}'.format(X_train.shape),'&','size of Y training set {}'.format(y_train.shape))
# print('Shape of X test set{}'.format(X_test.shape),'&','size of Y test set {}'.format(y_test.shape))

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
print(drugTree)
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)
print(predTree[0:5])
print(y_test[0:5])

from sklearn import metrics
import matplotlib.pyplot as plt

print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_test, predTree))

from io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

# import os
# os.environ["PATH"] += os.pathsep + 'D:/Graphviz 2.44.1/bin'

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
print(np.unique(y_train))

out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png(filename)
# img = mpimg.imread(filename)
# # plt.figure(figsize=(100, 200))
# plt.imshow(img,interpolation='nearest')
# plt.show()