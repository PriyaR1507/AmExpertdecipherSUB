"""
cubic lr 0.9982742257437094
quad lr 0.9962699891438975
xgb reg 0.9902314731106159
xgb reg 200 0.9779603890656204
xgb reg 300 0.9694051296088818
xgb reg 500 0.954436515095419
xgb reg 500-4 0.9210639235355089
xgb reg 0.8703834657089595
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, HuberRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.metrics import mean_squared_error
lgb = pickle.load(open('lgb','rb'))
huber = pickle.load(open('huber','rb'))
elastic = pickle.load(open('elastic','rb'))
ridge = pickle.load(open('ridge','rb'))
stack = pickle.load(open('stack','rb'))
svr = pickle.load(open('svr','rb'))
# = pickle.load(open('mlp'))

train_file = 'train_bFQbE3f/train.csv'
tee = pd.read_csv('test_9K3DBWQ.csv')
y =pd.read_csv(train_file)['cc_cons']
f = open('data6','rb')
data = pickle.load(f)
data.reset_index(drop=True,inplace=True)
train = data[:32820]
test = data[32820:]

cols=[]
for col in train.columns:
    if not (train[col] <0).any():
        cols.append(col)

train['cc_cons'] =y
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
train["cc_cons"] = np.log1p(train["cc_cons"])
y = train['cc_cons'].reset_index(drop=True)
y_avg = []
y_std = []
for i,j,k in zip(tee['cc_cons_apr'],tee['cc_cons_may'],tee['cc_cons_jun']):
    vals = [i,j,k]
    vals.sort()
    y_avg.append(0.5*vals[0]+0.5*vals[1])
    y_std.append(0.5*vals[0]+0.5*vals[2])

train_features = train.drop(['cc_cons'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)
skew_features = features[cols].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
X=features[:32820]
tX = features[32820:]
#print(X.shape)
#print(tX.shape)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#y_tr_lgb = lgb.predict(X)
y_te_lgb = lgb.predict(tX)
#print('lgb',y,y_tr_lgb)
#y_tr_huber = huber.predict(X)
y_te_huber = huber.predict(tX)
#print('huber',rmsle(y,y_tr_huber))
#y_tr_elastic= elastic.predict(X)
y_te_elastic= elastic.predict(tX)
#print('elastic',rmsle(y,y_tr_elastic))
#y_tr_ridge = ridge.predict(X)
y_te_ridge = ridge.predict(tX)
#print('ridge',rmsle(y,y_tr_ridge))
# y_tr_svr = svr.predict(X)
y_te_svr = svr.predict(tX)
# print('svr',rmsle(y,y_tr_svr))
#y_tr_stack = stack.predict(X)
y_te_stack = stack.predict(tX)
#print('stack',rmsle(y,y_tr_stack))

# tr_X= np.vstack([y_tr_lgb,y_tr_huber,y_tr_elastic,y_tr_ridge,y_tr_stack]).T
# te_X= np.vstack([y_te_lgb,y_te_huber,y_te_elastic,y_te_ridge,y_te_stack]).T
# tr_X = np.concatenate([X,tr_X],axis=1)
# te_X = np.concatenate([tX,te_X],axis=1)
# lightgbm = LGBMRegressor(objective='regression', 
#                                        num_leaves=8,
#                                        learning_rate=0.02, 
#                                        n_estimators=5000,
#                                        max_bin=200, 
#                                        bagging_fraction=0.75,
#                                        bagging_freq=5, 
#                                        bagging_seed=7,
#                                        feature_fraction=0.2,
#                                        feature_fraction_seed=7,
#                                        verbose=-1,
#                                        )
# alphas_alt = [3,3.5,3.6]
#alphas2 = [5e-05, 0.0001, 0.0003, 0.0004,0.0006, 0.0007, 0.0008]


# from sklearn.model_selection import GridSearchCV

# model = XGBRegressor(n_estimators=800,max_depth=3,learning_rate=0.1,reg_alpha=0.001)
# grd = GridSearchCV(model,param_grid={'learning_rate':[0.3,0.1,0.5],'n_estimators':[500,1000,2000,3000],
#                                     'reg_alpha':[0.001,0.0001,0.0005,0.005,0.008]},cv=3)
# grd.fit(tr_X,y)
# model.fit(tr_X,y)
# l2_y_tr = model.predict(tr_X)
# print(grd.best_params_)
# print(grd.best_score_)
# print("xgb reg",rmsle(y,l2_y_tr))
# print(np.expm1(l2_y_tr)[0:4])
# y_te = np.expm1(stack.predict(tX))
# sub=pd.read_csv('sub1.csv')
# sub['cc_cons'] = y_te
# sub.to_csv('sub3.csv',index=False)

# y_te = np.expm1(0.78*stack.predict(tX)+ 0.04*y_te_lgb+0.05*y_te_ridge+0.05*y_te_svr+0.03*y_te_elastic+0.03*y_te_huber)
sub=pd.read_csv('sub1.csv')
# sub['cc_cons'] = y_te
# sub.to_csv('sub12.csv',index=False)

# y_te = np.expm1(0.8*stack.predict(tX)+ 0.04*y_te_lgb+0.05*y_te_ridge+0.04*y_te_svr+0.035*y_te_elastic+
#                                 0.035*y_te_huber)
# sub['cc_cons'] = y_te
# sub.to_csv('sub12x.csv',index=False)
# sub=pd.read_csv('sub1.csv')
# sub['cc_cons'] = y_te
# sub.to_csv('sub12x1.csv',index=False)

# y_stk = np.square(np.expm1(pickle.load(open('y_te_data8_stack_3','rb'))))

# y_te = np.expm1(0.7*stack.predict(tX)+ 0.04*y_te_lgb+0.05*y_te_ridge+0.04*y_te_svr+0.035*y_te_elastic+
#                                 0.035*y_te_huber+0.2*np.log1p(y_stk))
# sub['cc_cons'] = y_te
# sub.to_csv('sub12x2.csv',index=False)

sub1=pd.read_csv('x1.csv')
sub2 = pd.read_csv('x2.csv')
sub['cc_cons'] = (sub1['cc_cons']+sub2['cc_cons'])/2
sub.to_csv('sub12x3.csv',index=False)




# y_te = np.expm1(0.8*stack.predict(tX)+ 0.08*y_te_lgb+0.12*y_te_ridge)
# sub=pd.read_csv('sub1.csv')
# sub['cc_cons'] = y_te
# sub.to_csv('sub32.csv',index=False)
###y_te = np.expm1(0.8*stack.predict(tX)+ 0.08*y_te_lgb+0.12*y_te_ridge)


# y_te = 0.95*np.expm1(0.8*stack.predict(tX)+0.08*y_te_ridge+0.12*y_te_ridge)+0.05*np.array(y_avg)
# sub=pd.read_csv('sub1.csv')
# sub['cc_cons'] = y_te
# sub.to_csv('sub3x.csv',index=False)
# y_te = 0.98*np.expm1(0.8*stack.predict(tX)+0.08*y_te_ridge+0.12*y_te_ridge)+0.02*np.array(y_avg)
sub=pd.read_csv('sub1.csv')
# sub['cc_cons'] = y_te
# sub.to_csv('sub3xx.csv',index=False)
