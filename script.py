import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold,GridSearchCV
import warnings
import pickle
warnings.filterwarnings('ignore')

train_file = 'train_bFQbE3f/train.csv'
test_file = 'test_9K3DBWQ.csv'

train =pd.read_csv(train_file)
test = pd.read_csv(test_file)

data =pd.concat([train,test])
data.drop(['id','cc_cons'],inplace=True,axis=1)

print("shape of data",data.shape)

def categorical_to_numeric(df):
    replace_dict = {'gender':{'M':0,'F':1},'account_type':{'saving':0,'current':1},
		    'loan_enq':{'Y':1}}
    df = df.replace(replace_dict)
    return df

def handle_missing_values(df):
    train= df[:32820]
    test = df[32820:].reset_index(drop=True)
    missing_cols = ['dc_cons_apr', 'dc_cons_may', 'dc_cons_jun', 'cc_count_apr',
       'cc_count_may', 'cc_count_jun', 'dc_count_apr', 'dc_count_may',
       'dc_count_jun', 'card_lim', 'personal_loan_active',
       'vehicle_loan_active', 'personal_loan_closed', 'vehicle_loan_closed',
       'investment_1', 'investment_2', 'investment_3', 'investment_4',
       'debit_amount_apr', 'credit_amount_apr', 'debit_count_apr',
       'credit_count_apr', 'max_credit_amount_apr', 'debit_amount_may',
       'credit_amount_may', 'credit_count_may', 'debit_count_may',
       'max_credit_amount_may', 'debit_amount_jun', 'credit_amount_jun',
       'credit_count_jun', 'debit_count_jun', 'max_credit_amount_jun',
       'loan_enq']
    for col in missing_cols:
        missing_col = 'missing%s'%col
        df[missing_col] = df[col].isnull().replace({False:0,True:1})
    return df


def feature_engg(data): 
    data = data.replace({'investment_1':{0.1:0},'investment_2':{0.1:0},'investment_3':{0.1:0},'investment_4':{0.1:0}})
    data.drop(['missinginvestment_1','missinginvestment_2','missinginvestment_3','missinginvestment_4'],axis=1,inplace=True)
    data = data.reset_index(drop=True)
    tinv=[]
    agg_inv = data['investment_1']+data['investment_2']+data['investment_3']+data['investment_4']
    for i in range(data.shape[0]):
        s = int(data['investment_1'].values[i]>0) + int(data['investment_2'].values[i]>0) +int(data['investment_3'].values[i]>0) +int(data['investment_4'].values[i]>0)
        if s==0:
            tinv.append(0)
            continue
        try:
            tinv.append(agg_inv[i]/s)
        except:
            print("yo",i)
            print(data.shape[0])
            pass
    data['agg_inv'] = tinv
    data['total_cc_cons'] = data['cc_cons_apr']+data['cc_cons_may']+data['cc_cons_jun']
    data.drop(['missingcredit_count_jun','missingmax_credit_amount_jun','missingdebit_amount_apr',
                'missingcredit_count_may','missingcredit_amount_jun','missingmax_credit_amount_may','missingdebit_count_may','missingcredit_amount_may',
                ],axis=1,inplace=True)
    def f(x):
        if x<0:
            return 0
        else:
            return x
    data['investment_4'] = data['investment_4'].apply(f)
    data['agg_inv'] =data['agg_inv'].apply(f)
    data['cc_apr_debit'] = (data['debit_amount_apr'] - data['cc_cons_apr']).apply(f) 
    data['cc_may_debit'] =(data['debit_amount_may'] - data['cc_cons_may']).apply(f)
    data['cc_jun_debit']=(data['debit_amount_jun'] - data['cc_cons_jun']).apply(f)

    per_cc_cons_apr = data['cc_cons_apr']/data['cc_count_apr']
    per_cc_cons_may = data['cc_cons_may']/data['cc_count_may']
    per_cc_cons_jun = data['cc_cons_jun']/data['cc_count_jun']

    data['total_credit_card_transactions'] = data['cc_count_apr']+data['cc_count_jun']+data['cc_count_may']
    data = data.drop(['cc_count_apr','cc_count_may','cc_count_jun'],axis=1)
    data['total_debit_card_transactions'] = data['dc_count_apr']+data['dc_count_jun']+data['dc_count_may']
    data = data.drop(['dc_count_apr','dc_count_may','dc_count_jun'],axis=1)
    data['both _active'] = [1 if i ==2 else 0 for i in data['personal_loan_active'] + data['vehicle_loan_active']]
    missing_per_cc_apr = []
    missing_per_cc_may = []
    missing_per_cc_jun = []
    for i in range(data.shape[1]):
        if per_cc_cons_apr[i]<0:
            per_cc_cons_apr[i] =-1
            missing_per_cc_apr[i] =1
        if per_cc_cons_may[i]<0:
            per_cc_cons_may =-1
            missing_per_cc_may[i] =1
        if per_cc_cons_jun[i]<0:
            per_cc_cons_jun=-1
            missing_per_cc_jun[i] =1 
        if per_cc_cons_apr[i]==np.inf:
            per_cc_cons_apr[i]=0
            missing_per_cc_apr[i] =1
        if per_cc_cons_may[i]==np.inf:
            per_cc_cons_may =0     
            missing_per_cc_may[i] =1
        if per_cc_cons_jun[i]==np.inf:
            per_cc_cons_jun=0
            missing_per_cc_jun[i] =1
    data['per_cc_apr'] = per_cc_cons_apr
    data['per_cc_may'] = per_cc_cons_may
    data['per_cc_jun'] = per_cc_cons_jun
    
    return data

#numeric and non-missing dat
data = feature_engg(handle_missing_values(categorical_to_numeric(data)).fillna(0.1))

with open('data12','wb') as f:
    pickle.dump(data,f)



