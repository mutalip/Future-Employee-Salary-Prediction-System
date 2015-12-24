import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

###############################################################################
# read data from excel file as DataFrame See more: http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html
raw_train_data = pd.read_excel("/Users/boyuan/Desktop/TrainingData.xlsx", parse_cols=[1,2,3,4,5,6,7,8,9,10,11])
raw_test_data = pd.read_excel("/Users/boyuan/Desktop/TestingData.xlsx", parse_cols=[1,2,3,4,5,6,7,8,9,10,11])

###############################################################################
# If the data has missing values, they will become NaNs in the resulting Numpy arrays.
# The vectorizer will create additional column <feature>=NA for each feature with NAs
raw_train_data = raw_train_data.fillna("NA")
raw_test_data = raw_test_data.fillna("NA")

exc_cols = [u'adjGross']
cols = [c for c in raw_train_data.columns if c not in exc_cols]

X_train = raw_train_data.ix[:,cols]
y_train = raw_train_data['adjGross'].values

X_test = raw_test_data.ix[:,cols]
y_test = raw_test_data['adjGross'].values

###############################################################################
# Convert DataFrame to dict See more: http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.to_dict.html#pandas.DataFrame.to_dict
dict_X_train = X_train.to_dict(orient='records')
dict_X_test = X_test.to_dict(orient='records')

###############################################################################
# 4.3.4 Encoding categorical features See more: http://scikit-learn.org/stable/modules/preprocessing.html
# 4.2.1 Loading features from dicts See more: http://scikit-learn.org/stable/modules/feature_extraction.html
# OneHotEncoder See more: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
# DictVectorizer See more : http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html

vec = DictVectorizer()
X_train = vec.fit_transform(dict_X_train).toarray()
X_test = vec.fit_transform(dict_X_test).toarray()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

score = clf.score(X_test,y_test)

print "Accuracy of the decision tree: ", score

###############################################################################
# export the tree as dot file

from sklearn.externals.six import StringIO

with open("tree_with_8_depth.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f, feature_names= vec.get_feature_names())


###############################################################################
import os
os.unlink('tree_with_8_depth.dot')
###############################################################################
import pydot 
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 

###############################################################################

from IPython.display import Image  
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
>>> Image(graph.create_png()) 
