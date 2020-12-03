#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV


# In[2]:


public_vacation_list = [
    '20170102', '20170128', '20170130', '20170131', '20170404',
    '20170414', '20170415', '20170417', '20170501', '20170503',
    '20170530', '20170701', '20171002', '20171005', '20171028',
    '20171225', '20171226', '20180101', '20180216', '20180217',
    '20180219', '20180330', '20180331', '20180402', '20180405',
    '20180501', '20180522', '20180618', '20180702', '20180925',
    '20181001', '20181017', '20181225', '20181226'
]


# In[3]:


training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")


# In[4]:


def preprocess_data(data):
    data_of_date = data['date'].tolist()
    date = []
    time = []
    for row in data_of_date:
        temp = row.split(" ")
        date.append(temp[0])
        time.append(int(temp[1].split(':')[0]))
    data['date'] = pd.to_datetime(date)
    data['time'] = time
    return data['date'],data['time']


# In[5]:


training_data['date'],training_data['time'] = preprocess_data(training_data)
training_data['year'] = training_data['date'].dt.year
training_data['month'] = training_data['date'].dt.month
training_data['day'] = training_data['date'].dt.day
training_data['dayofweek'] = training_data['date'].dt.dayofweek
training_data["weekofyear"] = training_data['date'].dt.week
training_data["quarter"] = training_data['date'].dt.quarter
training_data["dayofyear"] = training_data['date'].dt.dayofyear
training_data['date'] = training_data['date'].apply(lambda x: x.strftime('%Y%m%d'))
training_data['is_public_holiday'] = training_data['date'].apply(lambda x: 1 if x in public_vacation_list else 0)
training_data['is_weekend'] = training_data['dayofweek'].apply(lambda x: 1 if x == 5 or x == 6 else 0)


# In[6]:


training_data


# In[7]:


testing_data['date'],testing_data['time'] = preprocess_data(testing_data)
testing_data['year'] = testing_data['date'].dt.year
testing_data['month'] = testing_data['date'].dt.month
testing_data['day'] = testing_data['date'].dt.day
testing_data['dayofweek'] = testing_data['date'].dt.dayofweek
testing_data["weekofyear"] = testing_data['date'].dt.week
testing_data["quarter"] = testing_data['date'].dt.quarter
testing_data["dayofyear"] = testing_data['date'].dt.dayofyear
testing_data['date'] = testing_data['date'].apply(lambda x: x.strftime('%Y%m%d'))
testing_data['is_public_holiday'] = testing_data['date'].apply(lambda x: 1 if x in public_vacation_list else 0)
testing_data['is_weekend'] = testing_data['dayofweek'].apply(lambda x: 1 if x == 5 or x == 6 else 0)


# In[8]:


testing_data


# In[9]:


x_train = training_data.drop(["speed","date","id"], axis=1)
y_train = training_data["speed"]
x_test = testing_data.drop(["date","id"], axis=1)


# In[10]:


print(x_train)
print(y_train)
print(x_test)


# In[11]:


xx_train, xx_test, yy_train, yy_test = train_test_split(x_train, y_train,test_size=0.01)
dtrain = xgb.DMatrix(xx_train, label = yy_train)    
dvalid = xgb.DMatrix(xx_test, label = yy_test)      
dtest = xgb.DMatrix(x_test) 


# In[12]:


xgb_pars = {'objective': 'reg:squarederror', 'learning_rate': 0.02, 'min_child_weight': 0.7, 'max_depth': 10,  
            'subsample': 0.87, 'colsample_bytree': 1, 'colsample_bylevel': 0.68, 'reg_alpha': 0.2, 'gamma': 0.1,
            'reg_lambda': 0.4, 'nthread': 4}

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

model = xgb.train(xgb_pars, dtrain, 10000, watchlist, early_stopping_rounds=5,
      maximize=False, verbose_eval=1)

print('The RMSE is %.5f' % model.best_score)


# In[13]:


valid_pred = model.predict(dvalid)
print('valid_mse', mean_squared_error(yy_test,valid_pred))


# In[14]:


y_pred = model.predict(dtest)
result = []
for i in range(0, len(y_pred)):
    result.append([int(i), y_pred[i]])


# In[15]:


pd_data = pd.DataFrame(result, columns=['id', 'speed'])
pd_data.to_csv('submit.csv', index=None)


# In[ ]:




