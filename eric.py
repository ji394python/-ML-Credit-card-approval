import pandas as pd
from sklearn.decomposition import PCA

x_train = pd.read_csv('X_train.csv', index_col=0).drop(['member_id', 'id', 'pymnt_plan', 'policy_code', 'url'], axis=1)

# last_pymnt_d : 81 => 13 dim
# issue_d : 78 => 13 dim
# last_credit_pull_d : 81 => 13 dim
# earliest_cr_line : 758 => 13 dim
X_train['issue_d'] = X_train['issue_d'].apply(lambda x: str(x)[:3])
x_train['issue_d'] = x_train['issue_d'].apply(lambda x: str(x)[:3])
x_train['last_pymnt_d'] = x_train['last_pymnt_d'].apply(lambda x: str(x)[:3])
x_train['last_credit_pull_d'] = x_train['last_credit_pull_d'].apply(lambda x: str(x)[:3])
x_train['earliest_cr_line'] = x_train['earliest_cr_line'].apply(lambda x: str(x)[:3])

x_train_one_hot = pd.get_dummies(x_train[['grade','sub_grade','home_ownership','verification_status','purpose',
                                          'addr_state','initial_list_status','emp_length','application_type',
                                          'verification_status_joint','hardship_flag','hardship_type','hardship_reason',
                                          'hardship_status','hardship_loan_status','debt_settlement_flag',
                                          'settlement_status']], dummy_na=True)

x_train_one_hot = pd.concat([x_train_one_hot, pd.get_dummies(x_train[['issue_d','last_pymnt_d','last_credit_pull_d',
                                                                      'earliest_cr_line']])], axis=1)


x_train_one_hot = x_train_one_hot.astype('float')

x_test = pd.read_csv('X_test.csv', index_col=0).drop(['member_id', 'id', 'pymnt_plan', 'policy_code', 'url'], axis=1)

x_test['issue_d'] = x_test['issue_d'].apply(lambda x: str(x)[:3])
x_test['last_pymnt_d'] = x_test['last_pymnt_d'].apply(lambda x: str(x)[:3])
x_test['last_credit_pull_d'] = x_test['last_credit_pull_d'].apply(lambda x: str(x)[:3])
x_test['earliest_cr_line'] = x_test['earliest_cr_line'].apply(lambda x: str(x)[:3])

x_test_one_hot = pd.get_dummies(x_test[['grade','sub_grade','home_ownership','verification_status','purpose',
                                        'addr_state','initial_list_status','emp_length','application_type',
                                        'verification_status_joint','hardship_flag','hardship_type','hardship_reason',
                                        'hardship_status','hardship_loan_status','debt_settlement_flag',
                                        'settlement_status']], dummy_na=True)

x_test_one_hot = pd.concat([x_test_one_hot, pd.get_dummies(x_test[['issue_d','last_pymnt_d','last_credit_pull_d',
                                                                   'earliest_cr_line']])], axis=1)
x_test_one_hot = x_test_one_hot.astype('float')

# After Adding 4 date column
# 253, 250
# > 80% : 65
# > 90% : 88

pca = PCA(n_components=88)
pca_train = pca.fit_transform(x_train_one_hot)

pca = PCA(n_components=88)
pca_test = pca.fit_transform(x_test_one_hot)

print(pca_train.shape)
print(pca_test.shape)

import numpy as np
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix

# type 2 data : train_numeric_cleaned_large (delete 29 cols + only numeric + fillna)
train_numeric_cleaned_large = pd.read_csv('train_numeric_cleaned_large.csv')
y = train_numeric_cleaned_large['loan_status'].apply(lambda x : 0 if x == 'N' else 1).values
x = train_numeric_cleaned_large.drop(['loan_status'], axis=1).values

x_test = pd.read_csv('X_test.csv', index_col=0)
x_test['term'] = x_test['term'].apply(lambda x:str(x).replace(' months', '')).astype('float')
x_test['int_rate'] = x_test['int_rate'].apply(lambda x:str(x).replace('%', '')).astype('float')
x_test['revol_util'] = x_test['revol_util'].apply(lambda x:str(x).replace('%', '')).astype('float')
x_test = x_test[train_numeric_cleaned_large.columns.tolist()[:-1]]
x_test.fillna(x_test.median(), inplace=True)
y_test = pd.read_csv('Y_test.csv', index_col=0)

x_train_pca = np.append(x, pca_train, axis=1)
x_test_pca = np.append(x_test.values, pca_test, axis=1)

# StratifiedKFold
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_index, valid_index in skf.split(x_train_pca, y):
    x_train, x_valid = x_train_pca[train_index], x_train_pca[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

x_train_scaled = scaler.fit_transform(x_train) 
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test_pca)

print(x_train_scaled.shape)
print(x_valid_scaled.shape)
print(x_test_scaled.shape)