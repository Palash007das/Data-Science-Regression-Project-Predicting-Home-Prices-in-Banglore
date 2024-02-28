#!/usr/bin/env python
# coding: utf-8

# # Data Science Regression Project: Predicting Home Prices in Banglore

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # DATA EXPLOR

# In[2]:


df1=pd.read_csv("BHP.csv")
df1.head()


# In[3]:


df1.shape


# In[4]:


sns.countplot(x=df1.area_type,data=df1)
df1['area_type'].value_counts()


# Drop features that are not required to build our model

# In[5]:


df2=df1.drop(['area_type','availability','society','balcony'],axis=1)
df2.shape


# In[6]:


df2.head()


# In[7]:


df2.isnull().sum()


# # DATA CLENING

# In[8]:


df3=df2.dropna()


# In[9]:


df3.isnull().sum()


# In[14]:


df3['size'].unique()


# Add new feature(integer) for bhk (Bedrooms Hall Kitchen)

# In[15]:


df3['bhk']=df3['size'].apply(lambda x:int(x.split(" ")[0]))


# In[22]:


df3.head(3)


# In[23]:


df3.info()


# In[24]:


df3["total_sqft"].unique()


# In[25]:


def convert_sqft_to_avg(sqft):
    temp=sqft.split('-')
    if len(temp)==2:
        return (float(temp[0])+float(temp[1]))/2
    try:
        return float(sqft)
    except:
        return None
    
df3["total_sqft"]=df3["total_sqft"].apply(convert_sqft_to_avg)   


# In[26]:


df3.info()


# In[27]:


df3.isnull().sum()


# In[28]:


df4=df3.dropna()
df4.isnull().sum()


# In[29]:


df4.head()


# # FEATURE ENGINERING

# In[30]:


df4["price_per_sqft"]=df4['price']*100000/df4['total_sqft']


# In[31]:


df4.head()


# In[32]:


len(df4['location'].unique())#dimensionality curse or high dimensinality problem


# In[33]:


loc_cou=df4['location'].value_counts()
loc_cou


# In[34]:


loc_to_change=loc_cou[loc_cou<=20].index.tolist()
df4['location']=df4['location'].apply(lambda x:'other' if x in loc_to_change else x)


# In[35]:


len(df4['location'].unique())


# In[36]:


df4.head(10)


# # outlier removal

# In[37]:


df4[df4.total_sqft/df4.bhk<300].head()


# In[38]:


df4.shape


# In[39]:


df6=df4[~(df4.total_sqft/df4.bhk<300)]
df6.shape


# In[40]:


print(df6['price_per_sqft'].describe())
sns.boxenplot(df6.price_per_sqft)
plt.show()


# In[41]:


def remove_outlier(df):
   

    df_out = pd.DataFrame()

    
    for key, subdf in df.groupby('location'):
    
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)

        
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]

        
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)

    
    return df_out


df7 = remove_outlier(df6)


df7.shape


# In[42]:


df7['price_per_sqft'].describe()


# In[43]:


def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.bhk==2)]
    bhk3=df[(df.location==location)&(df.bhk==3)]
    plt.figure(figsize=(15,10))
    plt.scatter(bhk2.total_sqft,bhk2.price,color="b",label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color="r",label='3 BHK',s=50)
    plt.xlabel('Total sqft Area')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
plot_scatter_chart(df7,"Rajaji Nagar")    


# In[44]:


plot_scatter_chart(df7,"Hebbal") 


# In[45]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[46]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[47]:


plot_scatter_chart(df8,"Hebbal")


# In[48]:


plt.figure(figsize=(15,10))
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# # Outlier Removal Using Bathrooms Feature

# In[49]:


df8.bath.unique()


# In[50]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[51]:


df8[df8.bath>10]


# It is unusual to have 2 more bathrooms than number of bedrooms in a home

# In[52]:


df8[df8.bath>df8.bhk+2]


# Again the business manager has a conversation with you (i.e. a data scientist) that if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed

# In[53]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[54]:


df9.head(2)


# In[55]:


df10 = df9.drop(['size','price_per_sqft'],axis=1)
df10.head(3)


# # Use One Hot Encoding For Location

# In[56]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[57]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[58]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# # Build a Model Now...

# In[ ]:


df12.shape


# In[59]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[60]:


y = df12.price
y.head(3)


# In[61]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[62]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# # Use K Fold cross validation to measure accuracy of our LinearRegression model

# In[63]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# We can see that in 5 iterations we get a score above 80% all the time. This is pretty good but we want to test few other algorithms for regression to see if we can get even better score. We will use GridSearchCV for this purpose

# # Find best model using GridSearchCV

# In[64]:


from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Replace X and y with your actual data
# X = ...
# y = ...

find_best_model_using_gridsearchcv(X, y)


# Based on above results we can say that LinearRegression gives the best score. Hence we will use that.

# # Test the model for few properties

# In[65]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[66]:


print(predict_price('1st Phase JP Nagar',1000, 2, 2))
print(predict_price('1st Phase JP Nagar',1000, 3, 3))
print(predict_price('Indira Nagar',1000, 2, 2))
print(predict_price('Indira Nagar',1000, 3, 3))


# # Export the tested model to a pickle file

# In[ ]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# # Export location and column information to a file that will be useful later on in our prediction application

# In[ ]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




