# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, HuberRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
#from sklearn.ensemble import RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
#import seaborn as sns
#from sklearn.manifold import TSNE
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import pickle
import os


import warnings
warnings.filterwarnings('ignore')
train_file = 'train_bFQbE3f/train.csv'
y =pd.read_csv(train_file)['cc_cons']
print(y.shape)
import pickle
f = open('data12','rb')
data = pickle.load(f)
data.reset_index(drop=True,inplace=True)
train = data[:32820]
test = data[32820:]
#print(data.columns)
cols=[]
for col in train.columns:
    if not (train[col] <0).any():
        cols.append(col)
train['cc_cons'] =y
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
y = y.apply(np.sqrt).apply(np.log1p)
print(y.isnull().any())
y = train['cc_cons'].reset_index(drop=True)
train_features = train.drop(['cc_cons'], axis=1)
test_features = test
features = pd.concat([train, test_features]).reset_index(drop=True)
skew_features = features[cols].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
X=features[:32820]
tX = features[32820:]
kfolds = KFold(n_splits=3, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)
alphas_alt = [3,3.5,3.6]
#alphas2 = [5e-05, 0.0001, 0.0003, 0.0004,0.0006, 0.0007, 0.0008]
e_alphas = [ 0.0002, 0.0004, 0.0005]
# svr = make_pipeline(RobustScaler(), SVR(epsilon=0.001, kernel= 'linear'))
#huber = make_pipeline(StandardScaler(),HuberRegressor(alpha= 0.0009, epsilon=1.8, max_iter= 100))
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
# #lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
#elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds))
lr = make_pipeline(RobustScaler(),PolynomialFeatures(2),linear_model.LinearRegression())
lightgbm = LGBMRegressor(objective='regression', 
                                        num_leaves=6,
                                        learning_rate=0.3, 
                                        n_estimators=2000,
                                        max_bin=200, 
                                        bagging_fraction=0.75, 
                                        bagging_seed=7,
                                        feature_fraction=0.2,
                                        feature_fraction_seed=7,
                                        verbose=-1,
                                        )

stack_gen = StackingCVRegressor(regressors=(ridge,lr,lightgbm),
                                 meta_regressor=lightgbm,
                                 use_features_in_secondary=True)
# print('START Fit')
# print("huber")
# huber_model_full_data = huber.fit(np.array(X),np.array(y))

# print('stack_gen')
# stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

# print('elasticnet')
# elastic_model_full_data = elasticnet.fit(X, y)

# print('Lasso')
# lasso_model_full_data = lasso.fit(X, y)

# print('Ridge')
# ridge_model_full_data = ridge.fit(X, y)

# print('Svr')
# svr_model_full_data = svr.fit(X, y)

# print('xgboost')
# xgb_model_full_data = xgboost.fit(X, y)

# print('lightgbm')
# lgb_model_full_data = lightgbm.fit(X, y)   

# with open('lgb','wb') as f:
#     pickle.dump(lgb_model_full_data,f)

# with open('ridge','wb') as f:
#     pickle.dump(ridge_model_full_data,f)

# with open('elastic','wb') as f:
#     pickle.dump(elastic_model_full_data,f)

# with open('huber','wb') as f:
#      pickle.dump(huber_model_full_data,f)


stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
print('stack save')
y_te = stack_gen_model.predict(tX)
with open('y_te_data8_stack_6','wb') as f:
     pickle.dump(y_te,f)
with open('stack6','wb') as f:
     pickle.dump(stack_gen_model,f)

# print("svr")
# svr_model_full_data = svr.fit(np.array(X),np.array(y))
# with open('svr','wb') as f:
#     pickle.dump(svr_model_full_data,f)
# from sklearn.model_selection import GridSearchCV
# rf = RandomForestRegressor()
# rf_grid = GridSearchCV(rf,param_grid={'max_depth'=[5,10,15,20,25],'n_estimators':[100,200,500],
#                         'min_samples_split': [5,10,15, 20],"min_samples_leaf": [1,5, 10, 20],
#               "max_leaf_nodes": [15,20, 40],
#               "min_weight_fraction_leaf": [0.1]}
#                     })

# lightgbm = LGBMRegressor(objective='regression', 
#                                        learning_rate=0.01, 
#                                        n_estimators=5000,
#                                        verbose=-1,
#                                        )

# param_grid = {
#     'num_leaves': list(range(8, 30, 4)),
#     'min_data_in_leaf': [10, 20, 40, 60],
#     'max_depth': [3, 4, 5, 6, 8, -1],
#     'learning_rate': [0.1, 0.05, 0.01, 0.005],
#     'bagging_freq': [3, 4, 5, 6, 7],
#     'bagging_fraction': np.linspace(0.6, 0.95, 10),
#     'reg_alpha': np.linspace(0.1, 0.95, 10),
#     'reg_lambda': np.linspace(0.1, 0.95, 10),
#     'feature_fraction':[0.1,0.2,0.3],
#     'feature_fraction_seed':[5,7,8,10],
#     'bagging_seed':[7,12],
#     'max_bin':[100,150,200]
# }
# import gc
# gc.collect()

# lgb_grid = GridSearchCV(lightgbm,param_grid=param_grid,cv=kfolds)
# lgb_grid.fit(X,y)
# print(lgb_grid.best_params_)
# with open('lgb_grid','wb') as f:
#     pickle.dump(lgb_grid,f)
#import gc
#gc.collect()
# xgb_parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['reg:linear',],
#               'learning_rate': [0.1,.03, 0.05, .07], #so called `eta` value
#               'max_depth': [4,5, 6, 7,8],
#               'min_child_weight': [3,4,8],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.7],
#               'n_estimators': [2000,4000]}
# xgb = XGBRegressor()
# xgb_grid = GridSearchCV(xgb,param_grid=xgb_parameters,cv=kfolds)
# xgb_grid.fit(X.values,y)
# print(xgb_grid.best_params_)
# with open('xgb_grid','wb') as f:
#     pickle.dump(xgb_grid,f)

# stack_gen = StackingCVRegressor(regressors=(ridge,huber,elasticnet,lightgbm),
#                                 meta_regressor=lightgbm,
#                                 use_features_in_secondary=True)
