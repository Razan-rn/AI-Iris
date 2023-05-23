#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("IRIS.csv")
df.head()


# In[62]:


df.tail()#يعرض لي آخر البيانات


# In[63]:


df.info()#معلومات عن البيانات الي عندي


# In[10]:


df.isnull().sum()


# In[64]:


df.columns#اسماء العواميد


# In[2]:


df.rename(columns={"species":"class"},inplace=True)#يغيير لي اسم عمود
df.head(90)


# In[66]:


df.shape#كم صف وعمود


# In[67]:


#df.drop(['Id'],axis=1,inplace=True)


# In[3]:


df [df.duplicated(keep=False)]


# In[ ]:


#df.drop_duplicated(keep=Fales,inplace= True)


# In[11]:


df.describe()


# In[4]:


sns.boxplot(x='sepal_length',data=df)# يعرض لي البيانات على شكل مخططط واذا ابي عمود ثاني بس اغير الاسم


# In[12]:


sns.boxplot(x='sepal_width',data=df)#نفس الي فوق


# In[13]:


sns.boxplot(x='petal_length',data=df)#نفس الي فوق


# In[14]:


sns.boxplot(x='petal_width',data=df)#نفس الي فوق


# In[3]:


print(df.groupby('class').size())


# In[17]:


# histograms 
from matplotlib import pyplot
df.hist()
pyplot.show()


# In[4]:


#sns.boxplot(x='sepal_length')
df.plot(kind='box',subplots=True, layout=(2,2),sharex= False, sharey=False)
plt.show()


# In[71]:


from pandas.plotting import scatter_matrix
scatter_matrix(df)
plt.show()


# In[70]:


import seaborn as sns
sns.pairplot(df,markers=['o','s','d'],hue='class')
plt.show()                        


# In[18]:


feature_columns=['sepal_length','sepal_width','petal_length','petal_width']
x=df[feature_columns]
x


# In[19]:


y=df['class']
y


# In[20]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 4, shuffle = True)


# In[39]:


from sklearn.svm import SVC
def svm_model(x_train,y_train,x_test):
    svm = SVC(kernel='linear', C=1,random_state=0)
    svm.fit(x_train,y_train)
    y_pred=svm.predict(x_test)
    return y_pred


# In[40]:


y_pred=svm_model(x_train,y_train,x_test)
y_pred


# In[41]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(y_test, y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[42]:


from sklearn.linear_model import LogisticRegression
def LR_model(X_train,y_train,X_test):
    LR = LogisticRegression()
    LR.fit(X_train,y_train)
    y_pred=LR.predict(X_test)
    return y_pred


# In[43]:


y_pred=LR_model(x_train,y_train,x_test)
y_pred


# In[44]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(y_test, y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[45]:


from sklearn.neighbors import KNeighborsClassifier
def knn_model(X_train,y_train,X_test,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    return y_pred


# In[46]:


y_pred=knn_model(x_train,y_train,x_test,k=6)
y_pred


# In[47]:


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(y_test, y_pred)*100
print('Model Accuracy:',accuracy_sklearn)


# In[48]:


k_value=range(1,13)
accuracy=[]
for k in k_value:
    y_predict=knn_model(x_train,y_train,x_test,k)
    accur=accuracy_score(y_test,y_predict)
    accuracy.append(accur)


# In[49]:


plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(k_value,accuracy,c='g')
plt.show()


# In[ ]:




