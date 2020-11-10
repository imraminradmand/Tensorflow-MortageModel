#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')
df = pd.read_csv('../DATA/lending_club_loan_two.csv')


# In[3]:


df.info()


# In[4]:


sns.countplot(x='loan_status', data=df)


# unbalanced problem
# 
# Fraud and spam type data

# In[5]:


sns.distplot(df['loan_amnt'], kde=False, bins=42)


# Loans happening at standard amounts

# In[6]:


df.corr()


# In[7]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.ylim(10,0)


# Perfect corrolation along diag.
# 
# loan_amnt has good corrolation with installment

# In[8]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# Gives info on certain feature when called

# In[9]:


feat_info('installment')


# In[10]:


feat_info('loan_amnt')


# ^^ proves that 2 features are directly corrolated, which is obvious

# In[11]:


sns.scatterplot(x='installment', y='loan_amnt', data=df)


# Can see the corrolation more directly

# In[12]:


sns.boxplot(x='loan_status', y='loan_amnt', data=df)


# 
# Trying to determine if there is any relation between loans with higher amounts and lower amounts
# 
# ^^ not a good indicator to use

# In[13]:


df.groupby('loan_status')['loan_amnt'].describe()


# Shows numbers of above box plot for easier understanding

# In[14]:


df['grade'].unique()


# In[15]:


df['sub_grade'].unique()


# ^^ sub_grade hold actual grade in itself

# In[16]:


sns.countplot(x='grade', data=df,hue='loan_status')


# In[17]:


plt.figure(figsize=(14,4))
subgrade_order = sorted(df['sub_grade'].unique()) #reordering subgrades
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm',hue='loan_status')


# Lower grades have the similar given loan amount to write off rate, potentially not worth giving loans to that group?
# 
# F&G subgrades don't get paid back that well ==> isolate those

# In[18]:


f_and_g = df[(df['grade'] == 'G') |
            (df['grade'] == 'F')]

plt.figure(figsize=(14,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique()) 
sns.countplot(x='sub_grade', data=f_and_g, order=subgrade_order, palette='coolwarm',hue='loan_status')


# In[19]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off' : 0})


# In[20]:


df[['loan_repaid','loan_status']]


# Creating new column to see if loans are repaid or not

# In[21]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# which number features have the highest corrolation with the labels
# 
# dropping loan_repaid since its useless and will show perfect corrolation

# In[22]:


df.head()


# In[23]:


len(df)


# In[24]:


df.isnull().sum()


# finding out how many missing point there are in the data set

# In[25]:


df.isnull().sum() / len(df) * 100


# getting % of missing data

# In[26]:


feat_info('emp_title')
feat_info('emp_length')


# In[27]:


df['emp_title'].nunique()


# seeing how many unique titles we have
# 
# too many missing info to add dummy data

# In[28]:


df = df.drop('emp_title', axis=1)


# In[29]:


sorted(df['emp_length'].dropna().unique())


# In[30]:


emp_length_order = ['< 1 year',
 '1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years'
    
]


# In[31]:


plt.figure(figsize=(12,6))
sns.countplot(x='emp_length', data=df, order=emp_length_order,
             hue='loan_status')


# need to find ratio between blue and orange bars to get some more info (% charge of per cat.)

# In[32]:


emp_co = df[df['loan_status'] =='Charged Off'].groupby('emp_length').count()['loan_status']


# In[33]:


emp_fp = df[df['loan_status'] =='Fully Paid'].groupby('emp_length').count()['loan_status']


# In[34]:


emp_perc = emp_co / (emp_fp+emp_co)


# In[35]:


emp_perc.plot(kind='bar')


# no extreme differences to keep feature

# In[36]:


df = df.drop('emp_length', axis=1)


# In[37]:


df.isnull().sum()


# In[38]:


df['title'].head(10)


# In[39]:


df['purpose'].head(10)


# In[40]:


df = df.drop('title', axis=1)


# In[41]:


feat_info('mort_acc')


# In[42]:


df['mort_acc'].value_counts()


# how many mortgage accounts do people have in the dataset

# In[43]:


df.corr()['mort_acc'].sort_values()


# Trying to find a feature that would corrolate with mort_acc to fill in missing data from
# 
# could use total_acc to fill na

# In[44]:


df.groupby('total_acc').mean()['mort_acc']


# avg. mort_acc per total_acc
# 
# use this to fill missing data in mort_acc

# In[45]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[46]:


def fill_in_mort_acc(total_acc, mort_acc) :
    
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc] #look up average value for mort_acc based off total_acc
    else:
        return mort_acc


# function will look for account and if there is missing data it will fill it in from the avg. acc

# In[47]:


df['mort_acc'] = df.apply(lambda x: fill_in_mort_acc(x['total_acc'], x['mort_acc']), axis=1)


# In[48]:


df.isnull().sum()


# function worked, mort_acc is now 0 meaning there is no missing data
# 
# can drop other 2 rows as they are such a low %

# In[49]:


df = df.dropna()


# In[50]:


df.isnull().sum()


# everything looks good, no more null values

# In[51]:


df.select_dtypes(['object']).columns


# In[52]:


feat_info('term')


# In[53]:


df['term'].value_counts()


# In[54]:


df['term'] = df['term'].apply(lambda term: int(term[:3]))


# converting months to numerical values so it can be used

# In[55]:


df['term'].value_counts()


# In[56]:


df = df.drop('grade', axis = 1)


# dropping grade column since from earlier analysis found out sub_grade has grade in it, therefore its just useless and redundent

# In[57]:


dummies = pd.get_dummies(df[['sub_grade']],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1), dummies] , axis=1)


# convert sub_grade to dummies and drop the org. column to prevent variable mix up

# In[58]:


df.columns


# one-hot encoding was used

# In[59]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = pd.concat([df.drop(['verification_status', 'application_type','initial_list_status','purpose' ],axis=1), dummies] , axis=1)


# same thing as above, these columns were good candidates from the data analysis results

# In[60]:


df['home_ownership'].value_counts()


# In[61]:


df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')


# not enough values in none or any to be worth having them there, mapped them into other

# In[62]:


dummies = pd.get_dummies(df[['home_ownership']],drop_first=True)
df = pd.concat([df.drop('home_ownership',axis=1), dummies] , axis=1)


# In[63]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# In[64]:


df['zip_code'].value_counts()


# In[65]:


dummies = pd.get_dummies(df[['zip_code']],drop_first=True)
df = pd.concat([df.drop('zip_code',axis=1), dummies] , axis=1)


# In[66]:


df = df.drop('address', axis =1 )


# got the zip dont need the address anymore

# In[67]:


feat_info('issue_d')


# ^^ data leakage as we wont have an issue date for a new user and if we do that defeats the whole purpose of this model

# In[68]:


df = df.drop('issue_d', axis = 1)


# In[69]:


feat_info('earliest_cr_line')


# In[70]:


df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))


# Convert the date in earliest_cr_line to be useful

# In[72]:


df['earliest_cr_line']


# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


df = df.drop('loan_status',axis=1)


#   already got a loan_repaid column dont need this one anymore

# In[75]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[77]:


from sklearn.preprocessing import MinMaxScaler


# In[78]:


scaler = MinMaxScaler()


# In[79]:


X_train = scaler.fit_transform(X_train)


# In[80]:


X_test = scaler.transform(X_test)


# In[81]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


# In[82]:


model = Sequential()



model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# binary classification
model.add(Dense(units=1,activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam')


# In[83]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )


# In[84]:


from tensorflow.keras.models import load_model


# In[85]:


model.save('mortgagemodel.h5')


# In[86]:


losses = pd.DataFrame(model.history.history)


# In[ ]:


losses.plot()


# decent loss, not too much improvement, could add earlyStop and increasing epochs to improve learning

# In[89]:


from sklearn.metrics import classification_report,confusion_matrix


# In[90]:


prediction = model.predict_classes(X_test)


# In[91]:


print(classification_report(y_test, prediction))


# In[92]:


df['loan_repaid'].value_counts()


# In[93]:


317696 / len(df)


# ^^ accuracy of model before anything, therefore only improved by a small 9%

# pretty good on precision, lacking on recall, not too good of a f-1 score

# In[94]:


confusion_matrix(y_test, prediction)


# misclassifing a decent amount of 0 points

# In[96]:


import random
random.seed(34)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# Creating new customer to just test the model

# In[98]:


new_customer= scaler.transform(new_customer.values.reshape(1,78))


# In[99]:


model.predict_classes(new_customer)


# In[100]:


df.iloc[random_ind]['loan_repaid']


# Conclusion: model did predict the right outcome for the customer

# In[ ]:




