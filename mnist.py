# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:17:02 2016

@author: perry
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.grid_search import GridSearchCV
#from sklearn import cross_validation
#from sklearn.cross_validation import KFold
#from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
from datetime import datetime
import gc
import plotly
from plotly.graph_objs import Heatmap, Scatter, Contour

def value_counts(x):
    freq = pd.Series({val: sum(x == val) for val in set(x)})
    prop = freq / len(x)
    tbl = pd.concat([freq, prop], axis = 1)
    tbl.columns = ["freq", "prop"]
    return tbl




"""
Read Data
784 columns (28*28 pixels)
"""
train = pd.read_csv("D:/Users/perry/Downloads/train.csv", sep=",")
test = pd.read_csv("D:/Users/perry/Downloads/test.csv", sep=",")

train_x = train.ix[0:, 1:] / 255
train_y = train.ix[0:, 0]
test_x = train.ix[:,:] / 255




"""
Visualise column values
"""
plotly.offline.plot([Heatmap(x = np.arange(len(train)), y = np.arange(784), z = list(train.sort("label").apply(lambda x: list(x[1:] / 255), axis = 1)))])




"""
Method 1: Similarity
"""

label_avg = train.groupby("label").apply(np.mean).drop("label", axis = 1) / 255

#Function to calculate similarity metrics
cal_dist = lambda x: pd.Series([np.mean((np.array(x) - px)**2) for px in np.array(label_avg)])


##Prediction by shortest distance
train_x_sim = train_x.apply(cal_dist, axis = 1)
train_x_pred = train_x_sim.apply(lambda x: [d for d in range(10) if x[d] == np.min(x)][0], axis = 1)

value_counts(train_x_pred == train_y)
pd.crosstab(train_x_pred, train_y)








"""
Method 2: PCA then SVC
"""

pca100 = PCA(n_components=0.99, whiten=True)
pca100 = pca100.fit(train_x)
plotly.offline.plot([Scatter(x = np.arange(len(pca100.explained_variance_ratio_)), y = pca100.explained_variance_ratio_)])
plotly.offline.plot([Scatter(x = np.arange(len(pca100.explained_variance_ratio_)), y = np.cumsum(pca100.explained_variance_ratio_))])


#Arbitrary value 70%
pca70p = PCA(n_components=0.7, whiten=True)
train_pca70p_x = pca70p.fit_transform(train_x)
test_pca70p_x = pca70p.transform(test_x)

pca70p_svc_model = svm.SVC(kernel="rbf", C=1)
pca70p_svc_model_gsearch = GridSearchCV(estimator = pca70p_svc_model, param_grid = {'C': [1, 2, 5, 10, 20, 50, 100]}, cv=5)
pca70p_svc_model_gsearch.fit(train_pca70p_x, train_y)

#Grid search result
for x in pca70p_svc_model_gsearch.grid_scores_:
    print(x)

print(pca70p_svc_model_gsearch.best_score_)
print(pca70p_svc_model_gsearch.best_params_)
#mean: 0.97600, std: 0.00140, params: {'C': 1}
#mean: 0.97848, std: 0.00202, params: {'C': 2}
#mean: 0.97955, std: 0.00120, params: {'C': 5}
#mean: 0.97924, std: 0.00114, params: {'C': 10}
#mean: 0.97914, std: 0.00128, params: {'C': 20}
#mean: 0.97845, std: 0.00097, params: {'C': 50}
#mean: 0.97855, std: 0.00101, params: {'C': 100}

pca70p_svc_best_estimator = pca70p_svc_model_gsearch.best_estimator_

pd.DataFrame({'ImageId': np.arange(28000) + 1, 'Label': pca70p_svc_best_estimator.predict(test_pca70p_x)}).to_csv("D:/Users/perry/Downloads/pca70p_svc_gridsearch.csv", index = False)




#Arbitrary value 80%
pca80p = PCA(n_components=0.8, whiten=True)
train_pca80p_x = pca80p.fit_transform(train_x)
test_pca80p_x = pca80p.transform(test_x)

pca80p_svc_model = svm.SVC(kernel="rbf", C=1)
pca80p_svc_model_gsearch = GridSearchCV(estimator = pca80p_svc_model, param_grid = {'C': [1, 2, 5, 10, 20, 50, 100]}, cv=5)
pca80p_svc_model_gsearch.fit(train_pca80p_x, train_y)

#Grid search result
for x in pca80p_svc_model_gsearch.grid_scores_:
    print(x)

print(pca80p_svc_model_gsearch.best_score_)
print(pca80p_svc_model_gsearch.best_params_)
#mean: 0.97881, std: 0.00101, params: {'C': 1}
#mean: 0.98048, std: 0.00101, params: {'C': 2}
#mean: 0.98079, std: 0.00099, params: {'C': 5}
#mean: 0.98060, std: 0.00116, params: {'C': 10}
#mean: 0.98017, std: 0.00117, params: {'C': 20}
#mean: 0.98017, std: 0.00131, params: {'C': 50}
#mean: 0.98017, std: 0.00131, params: {'C': 100}

pca80p_svc_best_estimator = pca80p_svc_model_gsearch.best_estimator_

pd.DataFrame({'ImageId': np.arange(28000) + 1, 'Label': pca80p_svc_best_estimator.predict(test_pca80p_x)}).to_csv("D:/Users/perry/Downloads/pca80p_svc_gridsearch.csv", index = False)




#85%
pca85p = PCA(n_components=0.85, whiten=True)
train_pca85p_x = pca85p.fit_transform(train_x)
test_pca85p_x = pca85p.transform(test_x)

pca85p_svc_model = svm.SVC(kernel="rbf", C=1)
pca85p_svc_model_gsearch = GridSearchCV(estimator = pca85p_svc_model, param_grid = {'C': [1, 2, 5, 10, 20, 50, 100]}, cv=5)
pca85p_svc_model_gsearch.fit(train_pca85p_x, train_y)

#Grid search result
for x in pca85p_svc_model_gsearch.grid_scores_:
    print(x)

print(pca85p_svc_model_gsearch.best_score_)
print(pca85p_svc_model_gsearch.best_params_)
#mean: 0.97767, std: 0.00048, params: {'C': 1}
#mean: 0.98026, std: 0.00085, params: {'C': 2}
#mean: 0.98048, std: 0.00063, params: {'C': 5}
#mean: 0.98029, std: 0.00068, params: {'C': 10}
#mean: 0.98019, std: 0.00067, params: {'C': 20}
#mean: 0.98017, std: 0.00064, params: {'C': 50}
#mean: 0.98017, std: 0.00064, params: {'C': 100}

pca85p_svc_best_estimator = pca85p_svc_model_gsearch.best_estimator_

pd.DataFrame({'ImageId': np.arange(28000) + 1, 'Label': pca85p_svc_best_estimator.predict(test_pca85p_x)}).to_csv("D:/Users/perry/Downloads/pca85p_svc_gridsearch.csv", index = False)


plotly.offline.plot(
  [Contour(
    z = [[0.97600, 0.97848, 0.97955, 0.97924, 0.97914, 0.97845, 0.97855], 
         [0.97881, 0.98048, 0.98079, 0.98060, 0.98017, 0.98017, 0.98017], 
         [0.97767, 0.98026, 0.98048, 0.98029, 0.98019, 0.98017, 0.98017]], 
    x = [1, 2, 5, 10, 20, 50, 100], 
    y = [0.7, 0.8, 0.85]
  )]
)











"""
Method 3: xgboost
"""
param_search = {'max_depth': [2, 4, 6, 8],
                'min_child_weight': [1, 2, 3, 5, 8, 13]
                }


xgb_model = xgb.XGBClassifier(objective = "multi:softprob", gamma=0, max_depth=4, min_child_weight=5, n_estimators=400, learning_rate=0.05, subsample=0.8, nthread=4, seed=2016, silent=0)
xgb_model_gsearch = GridSearchCV(estimator = xgb_model, param_grid = param_search, cv=5)
xgb_model_gsearch.fit(train_x, train_y)

#Grid search result
for x in xgb_model_gsearch.grid_scores_:
    print(x)

print(xgb_model_gsearch.best_score_)
print(xgb_model_gsearch.best_params_)
#mean: 0.92748, std: 0.00315, params: {'min_child_weight': 1, 'max_depth': 2}
#mean: 0.92783, std: 0.00343, params: {'min_child_weight': 2, 'max_depth': 2}
#mean: 0.92783, std: 0.00239, params: {'min_child_weight': 3, 'max_depth': 2}
#mean: 0.92702, std: 0.00275, params: {'min_child_weight': 5, 'max_depth': 2}
#mean: 0.92781, std: 0.00218, params: {'min_child_weight': 8, 'max_depth': 2}
#mean: 0.92757, std: 0.00258, params: {'min_child_weight': 13, 'max_depth': 2}
#mean: 0.96655, std: 0.00146, params: {'min_child_weight': 1, 'max_depth': 4}
#mean: 0.96669, std: 0.00193, params: {'min_child_weight': 2, 'max_depth': 4}
#mean: 0.96602, std: 0.00154, params: {'min_child_weight': 3, 'max_depth': 4}
#mean: 0.96664, std: 0.00148, params: {'min_child_weight': 5, 'max_depth': 4}
#mean: 0.96586, std: 0.00192, params: {'min_child_weight': 8, 'max_depth': 4}
#mean: 0.96536, std: 0.00193, params: {'min_child_weight': 13, 'max_depth': 4}
#mean: 0.97279, std: 0.00111, params: {'min_child_weight': 1, 'max_depth': 6}
#mean: 0.97281, std: 0.00144, params: {'min_child_weight': 2, 'max_depth': 6}
#mean: 0.97257, std: 0.00102, params: {'min_child_weight': 3, 'max_depth': 6}
#mean: 0.97226, std: 0.00150, params: {'min_child_weight': 5, 'max_depth': 6}
#mean: 0.97129, std: 0.00163, params: {'min_child_weight': 8, 'max_depth': 6}
#mean: 0.97071, std: 0.00115, params: {'min_child_weight': 13, 'max_depth': 6}
#mean: 0.97226, std: 0.00103, params: {'min_child_weight': 1, 'max_depth': 8}
#mean: 0.97236, std: 0.00150, params: {'min_child_weight': 2, 'max_depth': 8}
#mean: 0.97236, std: 0.00129, params: {'min_child_weight': 3, 'max_depth': 8}
#mean: 0.97226, std: 0.00110, params: {'min_child_weight': 5, 'max_depth': 8}
#mean: 0.97179, std: 0.00113, params: {'min_child_weight': 8, 'max_depth': 8}
#mean: 0.97119, std: 0.00207, params: {'min_child_weight': 13, 'max_depth': 8}

xgb_best_estimator = xgb_model_gsearch.best_estimator_

pd.DataFrame({'ImageId': np.arange(28000) + 1, 'Label': xgb_best_estimator.predict(test_x)}).to_csv("D:/Users/perry/Downloads/xgb_gridsearch.csv", index = False)




def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

cvs_search = [merge_two_dicts(s[0], {'cvs': s[2]}) for s in xgb_model_gsearch.grid_scores_]
cvs_search = [{'min_child_weight': [s["min_child_weight"] for n in range(len(s["cvs"]))], 
               'max_depth': [s["max_depth"] for n in range(len(s["cvs"]))], 
               'cvs': s["cvs"]} for s in cvs_search]
cvs_search = [pd.DataFrame(s) for s in cvs_search]
cvs_search = pd.concat(cvs_search, ignore_index = True)

cvs_mean_data = cvs_search.groupby(["max_depth", "min_child_weight"]).apply(np.mean)
cvs_mean_data = cvs_mean_data.groupby(["max_depth"]).apply(lambda df: Scatter(x = df["min_child_weight"].values, y = df["cvs"], mode = "lines", name = "max_depth=" + str(df["max_depth"].values[0])))
cvs_mean_data = list(cvs_mean_data)


cvs_data = cvs_search.groupby("max_depth").apply(lambda df: Scatter(x = df["min_child_weight"], y = df["cvs"], mode = "markers", name = "max_depth=" + str(df["max_depth"].values[0])))
cvs_data = list(cvs_data)


plotly.offline.plot(cvs_data + cvs_mean_data)
plotly.offline.plot(cvs_data[1:] + cvs_mean_data[1:])

pd.crosstab(xgb_model_gsearch.best_estimator_.predict(train_x), train_y)








"""
Method 3.2: PCA then xgboost
"""

#Arbitrary value 80%
pca80p = PCA(n_components=0.8, whiten=True)
train_pca80p_x = pca80p.fit_transform(train_x)
test_pca80p_x = pca80p.transform(test_x)

param_search = {'max_depth': [4, 6, 8, 10, 12, 14, 16],
                'min_child_weight': [1, 2, 3, 5, 8, 13]
                }

xgb_model = xgb.XGBClassifier(objective = "multi:softprob", gamma=0, max_depth=4, min_child_weight=5, n_estimators=400, learning_rate=0.05, subsample=0.8, nthread=4, seed=2016, silent=0)
xgb_model_gsearch = GridSearchCV(estimator = xgb_model, param_grid = param_search, cv=5)
xgb_model_gsearch.fit(train_pca80p_x, train_y)

#Grid search result
for x in xgb_model_gsearch.grid_scores_:
    print(x)

print(xgb_model_gsearch.best_score_)
print(xgb_model_gsearch.best_params_)
#mean: 0.90781, std: 0.00321, params: {'min_child_weight': 1, 'max_depth': 2}
#mean: 0.90824, std: 0.00357, params: {'min_child_weight': 2, 'max_depth': 2}
#mean: 0.90802, std: 0.00333, params: {'min_child_weight': 3, 'max_depth': 2}
#mean: 0.90833, std: 0.00347, params: {'min_child_weight': 5, 'max_depth': 2}
#mean: 0.90793, std: 0.00305, params: {'min_child_weight': 8, 'max_depth': 2}
#mean: 0.90838, std: 0.00335, params: {'min_child_weight': 13, 'max_depth': 2}
#mean: 0.94924, std: 0.00151, params: {'min_child_weight': 1, 'max_depth': 4}
#mean: 0.94921, std: 0.00166, params: {'min_child_weight': 2, 'max_depth': 4}
#mean: 0.94874, std: 0.00162, params: {'min_child_weight': 3, 'max_depth': 4}
#mean: 0.94867, std: 0.00214, params: {'min_child_weight': 5, 'max_depth': 4}
#mean: 0.94810, std: 0.00227, params: {'min_child_weight': 8, 'max_depth': 4}
#mean: 0.94774, std: 0.00219, params: {'min_child_weight': 13, 'max_depth': 4}
#mean: 0.96019, std: 0.00160, params: {'min_child_weight': 1, 'max_depth': 6}
#mean: 0.96012, std: 0.00144, params: {'min_child_weight': 2, 'max_depth': 6}
#mean: 0.95995, std: 0.00112, params: {'min_child_weight': 3, 'max_depth': 6}
#mean: 0.95940, std: 0.00181, params: {'min_child_weight': 5, 'max_depth': 6}
#mean: 0.95917, std: 0.00207, params: {'min_child_weight': 8, 'max_depth': 6}
#mean: 0.95783, std: 0.00165, params: {'min_child_weight': 13, 'max_depth': 6}
#mean: 0.96069, std: 0.00125, params: {'min_child_weight': 1, 'max_depth': 8}
#mean: 0.96026, std: 0.00116, params: {'min_child_weight': 2, 'max_depth': 8}
#mean: 0.96048, std: 0.00122, params: {'min_child_weight': 3, 'max_depth': 8}
#mean: 0.96081, std: 0.00155, params: {'min_child_weight': 5, 'max_depth': 8}
#mean: 0.96088, std: 0.00148, params: {'min_child_weight': 8, 'max_depth': 8}
#mean: 0.95983, std: 0.00085, params: {'min_child_weight': 13, 'max_depth': 8}
#mean: 0.95943, std: 0.00140, params: {'min_child_weight': 1, 'max_depth': 10}
#mean: 0.95971, std: 0.00149, params: {'min_child_weight': 2, 'max_depth': 10}
#mean: 0.96083, std: 0.00153, params: {'min_child_weight': 3, 'max_depth': 10}
#mean: 0.96079, std: 0.00162, params: {'min_child_weight': 5, 'max_depth': 10}
#mean: 0.96095, std: 0.00114, params: {'min_child_weight': 8, 'max_depth': 10}
#mean: 0.96005, std: 0.00171, params: {'min_child_weight': 13, 'max_depth': 10}
#mean: 0.95907, std: 0.00172, params: {'min_child_weight': 1, 'max_depth': 12}
#mean: 0.95960, std: 0.00186, params: {'min_child_weight': 2, 'max_depth': 12}
#mean: 0.95995, std: 0.00157, params: {'min_child_weight': 3, 'max_depth': 12}
#mean: 0.96033, std: 0.00184, params: {'min_child_weight': 5, 'max_depth': 12}
#mean: 0.96064, std: 0.00156, params: {'min_child_weight': 8, 'max_depth': 12}
#mean: 0.96012, std: 0.00159, params: {'min_child_weight': 13, 'max_depth': 12}
#mean: 0.95862, std: 0.00177, params: {'min_child_weight': 1, 'max_depth': 14}
#mean: 0.95948, std: 0.00154, params: {'min_child_weight': 2, 'max_depth': 14}
#mean: 0.95979, std: 0.00110, params: {'min_child_weight': 3, 'max_depth': 14}
#mean: 0.96081, std: 0.00130, params: {'min_child_weight': 5, 'max_depth': 14}
#mean: 0.96093, std: 0.00165, params: {'min_child_weight': 8, 'max_depth': 14}
#mean: 0.96029, std: 0.00159, params: {'min_child_weight': 13, 'max_depth': 14}
#mean: 0.95819, std: 0.00164, params: {'min_child_weight': 1, 'max_depth': 16}
#mean: 0.95979, std: 0.00152, params: {'min_child_weight': 2, 'max_depth': 16}
#mean: 0.96005, std: 0.00098, params: {'min_child_weight': 3, 'max_depth': 16}
#mean: 0.96090, std: 0.00155, params: {'min_child_weight': 5, 'max_depth': 16}
#mean: 0.96086, std: 0.00245, params: {'min_child_weight': 8, 'max_depth': 16}
#mean: 0.96036, std: 0.00100, params: {'min_child_weight': 13, 'max_depth': 16}

plotly.offline.plot(
  [Contour(
    z = [[0.90781, 0.90824, 0.90802, 0.90833, 0.90793, 0.90838], 
         [0.94924, 0.94921, 0.94874, 0.94867, 0.94810, 0.94774], 
         [0.96019, 0.96012, 0.95995, 0.95940, 0.95917, 0.95783], 
         [0.96069, 0.96026, 0.96048, 0.96081, 0.96088, 0.95983], 
         [0.95943, 0.95971, 0.96083, 0.96079, 0.96095, 0.96005], 
         [0.95907, 0.95960, 0.95995, 0.96033, 0.96064, 0.96012], 
         [0.95862, 0.95948, 0.95979, 0.96081, 0.96093, 0.96029], 
         [0.95819, 0.95979, 0.96005, 0.96090, 0.96086, 0.96036]], 
    x = [1, 2, 3, 5, 8, 13], 
    y = [2, 4, 6, 8, 10, 12, 14, 16]
  )]
)


