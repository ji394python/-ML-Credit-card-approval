from math import nan
import pandas as pd 
from datetime import datetime
import sklearn as sk
import numpy as np
import math
from sklearn import preprocessing

def LabelEncoder(df:pd.DataFrame, col_list):
    le_class_dict = {}
    for col in col_list:
        df[col].fillna('nan', inplace=True)
        print('LabelEncoder:',col)
        df[col] = df[col].astype('str')
        le = preprocessing.LabelEncoder()
        le.fit(df[col])
        le_class_dict[col] = le.classes_
        df[col] = le.transform(df[col])
        #將標籤化的字典儲存以利預測時使用
        np.save(f'encoder_json/{col}.npy', le_class_dict)
    return df 
            
#訓練使用
#將格式例如20200418的資料切割成年跟月，變成2020與04，再套用到Label Encoder.
#Author : Arleigh Chang
def date_data_convert(df:pd.DataFrame,columns,format):
    for column in columns:
        date = []
        print('date_data_convert,Dealing:',column) 
        for i in df[column] :
            try: 
                date.append(str(datetime.strptime(i,format))[5:10].split('-'))
            except:
                date.append(['99','99'])
        df[column+'month'] = [ m[0] for m in date]
        df[column+'date'] = [ d[1] for d in date]
        #因只需用到轉換完後的欄位，所以在此將原生欄位drop掉
        df.drop([column],axis=1,inplace=True)
        #套用label encoder
        df = LabelEncoder(df, [column+'month',column+'date'])  
    return df

### 讀取資料
X_train = pd.read_csv('X_train.csv').iloc[:,1:]
X_test = pd.read_csv('X_test.csv').iloc[:,1:]


### 比例處理
colProportion = ['revol_util','int_rate']
for i in colProportion:
    X_train[i].fillna('0',inplace=True)
    X_test[i].fillna('0',inplace=True)
    print('data_ratio_convert:Dealing',i)
    X_train[i] = X_train[i].str.replace('%','').astype(float)
    X_test[i] = X_test[i].str.replace('%','').astype(float)
    
### 日期處理
colDate = ['issue_d','earliest_cr_line','last_pymnt_d'
              ,'next_pymnt_d','last_credit_pull_d','hardship_start_date'
              ,'hardship_end_date','payment_plan_start_date','debt_settlement_flag_date','settlement_date']
X_train = date_data_convert(X_train,colDate,'%b-%d')
X_test = date_data_convert(X_test,colDate,'%b-%d')

'''
### 特殊欄位處理
colSpecail = ['desc','title'] #desc/title turn to [0,1]
for col in colSpecail:
    print('data_01_convert:Dealing',col)
    X_train[col] = [ 0 if v is None else 1 for v in X_train[col].values]
    X_test[col] = [ 0 if v is None else 1 for v in X_test[col].values]
'''

colDropDate = ['sec_app_earliest_cr_line','earliest_cr_line']
colDropOthers = ['emp_title','url','desc','member_id', 'id', 'pymnt_plan', 'policy_code','title']
colDrop = colDropDate + colDropOthers


for column in colDrop:
    print('dropCol,Dealing:',column)
    X_train.drop([column],axis=1,inplace=True)
    X_test.drop([column],axis=1,inplace=True)


colNormal = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status'
             , 'purpose', 'zip_code', 'addr_state', 'initial_list_status'
             , 'application_type', 'verification_status_joint', 'hardship_flag'
             , 'hardship_type', 'hardship_reason', 'hardship_status'
             , 'hardship_loan_status', 'debt_settlement_flag', 'settlement_status']

X_train = LabelEncoder(X_train,colNormal)
X_test = LabelEncoder(X_test,colNormal)