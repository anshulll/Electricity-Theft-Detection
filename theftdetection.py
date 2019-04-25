
# coding: utf-8

# In[172]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RB
#from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE as SM
from sklearn.ensemble import RandomForestRegressor as RF,GradientBoostingRegressor as GB,ExtraTreesClassifier as ET
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neural_network import MLPClassifier as MP


# In[2]:


df=pd.read_csv('data.csv')


# In[3]:


#df1=df[df["FLAG"]==0]


# In[4]:


#df1=df
df.head()


# In[5]:


math.isnan(df.iloc[0][2])


# In[7]:


#df1.shape


# In[8]:


#df.head(


# In[9]:


#df1.isnull().sum()
l=df.columns
la=['CONS_NO','FLAG']
lb=[]
for i in l:
    if i not in la:
        lb.append(i)


# In[10]:


import datetime
dates = [datetime.datetime.strptime(ts, "%Y/%m/%d") for ts in lb]
#dates.sort()
fdates = [datetime.datetime.strftime(ts, "%Y/%m/%d") for ts in dates]


# In[11]:


fdates.insert(0,"FLAG")
fdates.insert(0,"CONS_NO")


# In[12]:


df.columns=fdates


# In[13]:


import datetime
dates = [datetime.datetime.strptime(ts, "%Y/%m/%d") for ts in lb]
dates.sort()
sorteddates = [datetime.datetime.strftime(ts, "%Y/%m/%d") for ts in dates]


# In[14]:


cols=df.columns.tolist()[0:2]+sorteddates
df=df[cols]


# In[15]:


df


# In[16]:


#df=df[cols]
df1=df


# In[17]:


l=df["2014/01/01"]
l1=df["2014/01/02"]
l=np.asarray(l).tolist()
l1=np.asarray(l1).tolist()


# In[18]:


l2=[]
for i in range(len(l)):
    if math.isnan(l[i]):
        if math.isnan(l1[i]):
            l2.append(0)
        else:
            l2.append(l1[i]/2)
    else:
        l2.append(l[i])
df1["2014/01/01"]=l2


# In[19]:


df1.head()


# In[20]:


df.columns[-1],df.columns[-2]


# In[21]:


df.iloc[0][1035]


# In[22]:


l=df["2016/10/31"]
l1=df["2016/10/30"]
l=np.asarray(l).tolist()
l1=np.asarray(l1).tolist()


# In[23]:


l2=[]
for i in range(len(l)):
    if math.isnan(l[i]):
        if math.isnan(l1[i]):
            l2.append(0)
        else:
            l2.append(l1[i]/2)
    else:
        l2.append(l[i])
df1["2016/10/31"]=l2


# In[24]:


df1.head()


# In[25]:


#for i in range(42372):
#    l=np.asarray(df1[i:i+1]).tolist()[0]
#    l1=[]
#    l1[0:3]=l[0:3]
#    for j in range(3,1035):
#        if math.isnan(l[j]):
#            if math.isnan(l[j+1])==False and math.isnan(l[j-1])==False:
#                l1.append((l[j-1]+l[j+1])/2)
#            else:
#                l1.append(0)
#        else:
#            l1.append(l[j])
#    l1.append(l[-1])
#   df1[i:i+1]=l1


# In[26]:


df.columns


# In[142]:


#df1.isnull().sum()
l=df.columns
la=['CONS_NO','FLAG']
lbx=[]
for i in l:
    if i not in la:
        lbx.append(i)


# In[28]:


lb


# In[29]:


for i in range(1,len(lb)-1):
    l=np.asarray(df[lb[i]]).tolist()
    l1=np.asarray(df[lb[i-1]]).tolist()
    l2=np.asarray(df[lb[i+1]]).tolist()
    l3=[]
    for j in range(len(l)):
        if math.isnan(l[j]):
            if math.isnan(l1[j])==False and math.isnan(l2[j])==False:
                l3.append((l1[j]+l2[j])/2)
            else:
                l3.append(0)
        else:
            l3.append(l[j])
    df1[lb[i]]=l3


# In[30]:


df1.head()


# In[31]:


#plt.scatter(df["2014/1/14"],df["FLAG"])


# In[130]:


#df.isnull().sum(axis=1)
#df1.iloc[4].value_counts()


# In[138]:


df1.iloc[3800]["FLAG"]


# In[143]:


l1=lbx[100:990]


# In[150]:


for i in range(5700,5710):
    plt.plot(l1,df1.iloc[i][l1])
    #plt.show()


# In[153]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(24,16), dpi=200, facecolor='w', edgecolor='k')
axes = plt.gca()
fig = plt.gcf()
#axes.set_xlim([,])
#x1,x2,y1,y2 = plt.axis()
#plt.axis((x1,x2,0,50))
#axes.set_ylim([0,50])
for i in range(1700,1710):
    plt.plot(l1,df1.iloc[i][l1],c='r')
    plt.plot(l1,df1.iloc[i+4000][l1],c='b')
    #plt.plot(l1,)
    #plt.show()
#fig.set_size_inches(18.5, 10.5)
fig.savefig('wihoutcluster.png', dpi=200)


# In[35]:


#4372,14,4919,4808,261 278 267 280 must
#major start from col 700 to last
#on whole a constancy should be maintained for not theft
axes = plt.gca()
#axes.set_xlim([,])
axes.set_ylim([0,50])
for i in range(5700,5710):
    plt.plot(l1,df1.iloc[i][l1])
    plt.show()


# In[36]:


for i in range(250,300):
    plt.plot(l1,df1.iloc[i][l1])


# In[37]:


#c=0
#for i in l:
#    if(i<=70):
#        c+=1
#c
#df.iloc[4372].isnull().sum()
axes = plt.gca()
axes.set_ylim([0,250])
for i in range(250,300):
    plt.plot(l1,df1.iloc[i][l1])
    plt.show()


# In[38]:




#df.isnull().sum(axis=1)
#may be theft in a period of time not always


# In[39]:


#l1=[]
#for i in range(1000):
#    l=np.asarray(df1.iloc[i][lb]).tolist()
#    x=statistics.stdev(l)
#    y=statistics.mean(l)
#    l1.append(y+2*x)
    #df1.iloc[i][lb].apply(lambda t:y+2*x if t>y+2*x else t)


# In[40]:


#df.iloc[1].isnull().sum()
#dfw=df[df["means"]>=1]
#l=df.columns.tolist()
#l[970]


# In[41]:


#df1.iloc[1].describe()


# In[42]:


#l=[df1.iloc[i].value_counts for i in range(42372)]
#df1.iloc[0].count(0)
#l=(df1==0).astype(int).sum(axis=1)


# In[43]:


#l[250]


# In[44]:


#df["means"]=df.mean(axis=1,skipna=True)
#la=df.mean(axis=1,skipna=True)
#l
#c=0
#for i in l:
#    if i<=15000:
#        c+=1
#c


# In[180]:


#c=0
#for i in range(len(l)):
#    if l[i]<=200:
#        c+=1
#c
#df["mean"].shape
#df.fillna(df.means)


# In[46]:


#for i in range(0,len(lb)):
#    l=np.asarray(df[lb[i]]).tolist()
    #l1=np.asarray(df[lb[i-1]]).tolist()
    #l2=np.asarray(df[lb[i+1]]).tolist()
#    l3=[]
#    for j in range(len(l)):
#        if math.isnan(l[j]):
#            l3.append(la[j])
#        else:
#            l3.append(l[j])
#    df[lb[i]]=l3


# In[47]:


#df1["zero"]=l


# In[48]:


#df2=df1[df1["zero"]<=400]


# In[49]:


#df2["FLAG"].value_counts()


# In[155]:


#len(X)
len(lbx)


# In[156]:


#l=np.asarray(df1.iloc[0][lb]).tolist()
#x=statistics.stdev(d)
#y=statistics.mean(l)
#d=df1.iloc[0][lb].apply(lambda t:y+2*x if t>y+2*x else t)
X=df1[lbx[500:]]
Y=df1["FLAG"]


# In[160]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)


# In[161]:


sm=SM(random_state=42)
#X_tr,Y_tr=sm.fit_sample(X_train,Y_train)
X_tr,Y_tr=sm.fit_sample(X_train,Y_train)


# In[162]:


#model=RB()
#model.fit(X_trainpca,Y_train)
model=ET(n_estimators=10)
model.fit(X_tr,Y_tr)


# In[163]:


model.score(X_test,Y_test)


# In[164]:


l1=model.predict(X_test)
f1_score(Y_test,l1)


# In[166]:


Y_test=Y_test.tolist()


# In[170]:


c=0
for i in range(len(l1)):
    if(Y_test[i])==0:
        c+=1
c


# In[171]:


c=0
for i in range(len(l1)):
    if l1[i]!=Y_test[i] and Y_test[i]==0:
        c+=1
print(c)


# In[53]:


#pca = PCA(n_components=50).fit(X)
clf=KMeans(n_clusters=10)
clf.fit(X)


# In[54]:


#df.iloc[4919].isnull().sum()
#X_pca= pca.transform(X)
centroids=clf.cluster_centers_
labels=clf.labels_
colors=["g.","r.","c.","b.","k.","o."]


# In[55]:


t=clf.predict(X)


# In[56]:


t=t.tolist()


# In[57]:


c=0
l=[]
for i in range(len(t)):
    if Y[i]==1:
        l.append(t[i])


# In[58]:


c=0
for i in l:
    if i==0:
        c+=1
c


# In[60]:


df1["labels"]=labels


# In[61]:


X["labels"]=labels


# In[62]:


X1=X[X["labels"]==0]


# In[63]:


X1=X1.drop(["labels"],axis=1)


# In[64]:


l=[]
for i in range(41809):
    l.append(X1.iloc[i]) 


# In[65]:


#c=0
#l1=[]
#for i in l:
#    for j in range(534):
#        c+=(i[j]-centroids[4][j])**2
#    c=np.sqrt(c)
#    l1.append(c)
#    c=0


# In[66]:


t1=list((np.array(l) - np.array(centroids[0]))**2)


# In[67]:


l2=[]
for i in t1:
    l2.append(np.sqrt(np.sum(i)))


# In[68]:


mean=np.mean(l2)


# In[69]:


Y1=Y[X["labels"]==0]


# In[70]:


Y1=Y1.tolist()


# In[71]:


c=0
l3=[]
for i in range(41809):
    if Y1[i]==1:
        l3.append(l2[i])


# In[72]:


#out of 3371  2716 with distance greater than 100.
X1["distance"]=l2


# In[73]:


X1["theft"]=Y1


# In[74]:


X2=X1
len(l2)


# In[75]:


c=0
for i in range(len(l2)):
    if l2[i]<80 and Y1[i]==0:
        c+=1
c


# In[76]:


c=0
h=[]
for i in range(len(l2)):
    f=0
    if (l2[i]<=70 and Y1[i]==0) or (l2[i]>100 and Y1[i]==1):
        f=1
    if f==0:
        h.append(int(i))
#X2=X2.drop(h,axis=1)


# In[77]:


X3=X2.drop(X2.index[h])


# In[78]:


#Yf=X3["theft"]
#X3=X3.drop("theft",axis=1)
Ecd=X3["distance"]
X3=X3.drop("distance",axis=1)


# In[79]:


lb=X3.columns.tolist()


# In[191]:


l1=lb


# In[81]:


#Yf=Yf.tolist()
Yf=X3["theft"]


# In[82]:


Yf=Yf.tolist()


# In[83]:


c=0
for i in Yf:
    if i==0:
        c+=1
c


# In[84]:


len(Yf)


# In[196]:


Yf[1000]
#len(l1)


# In[86]:


#figure(num=None, figsize=(32,24), dpi=80, facecolor='w', edgecolor='k')
plt.plot(l1,X3.iloc[3013][l1])


# In[195]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(32,24), dpi=80, facecolor='w', edgecolor='k')
axes = plt.gca()
#axes.set_xlim([,])
#x1,x2,y1,y2 = plt.axis()
#plt.axis((x1,x2,0,50))
#axes.set_ylim([0,50])
for i in range(3000,3005):
    plt.plot(l1,X3.iloc[i][l1],c='b')
    plt.plot(l1,X3.iloc[i-2000][l1],c='r')
    #plt.plot(l1,)
    #plt.show()


# In[128]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(24,16), dpi=200, facecolor='w', edgecolor='k')
axes = plt.gca()
fig = plt.gcf()
#axes.set_xlim([,])
#x1,x2,y1,y2 = plt.axis()
#plt.axis((x1,x2,0,50))
#axes.set_ylim([0,50])
for i in range(2000,2010):
    plt.plot(l1,X3.iloc[i][l1],c='r')
    plt.plot(l1,X3.iloc[i+1000][l1],c='k')
    #plt.plot(l1,)
    #plt.show()
#fig.set_size_inches(18.5, 10.5)
fig.savefig('boththeft1.png', dpi=200)


# In[89]:


Ecd=Ecd.tolist()


# In[90]:


Ecd[2008]
#ecd less than 70 is 0 >100 is 1


# In[91]:


## l3=[]
l4=[]
for i in range(len(l2)):
    if l2[i]<=70 and Y1[i]==0:
        l3.append(X1.iloc[i])
    elif l2[i]>=100 and Y1[i]==1:
        l4.append(X1.iloc[i])


# In[92]:


#h=[[2,3],[4,5]]
#g=[3,4]
#t2=list(([np.array(i) for i in h] - np.array(g))**2)
#t2=list((np.array([2,3])-np.array([3,4])))


# In[181]:


X_train,X_test,Y_train,Y_test=train_test_split(X3,Yf,test_size=0.20)


# In[183]:


sm=SM(random_state=42)
#X_tr,Y_tr=sm.fit_sample(X_train,Y_train)
X_tr,Y_tr=sm.fit_sample(X_train,Y_train)


# In[182]:


#statistics.stdev(l)
#X_trainpca.shape
#transformer = KernelPCA(n_components=50, kernel='linear')
#X_transformed =transformer.fit_transform(X_train)
X_train.shape


# In[96]:


#l1=df1.isnull().sum(axis=1)
#l1=np.asarray(l1).tolist()


# In[184]:


#model=RB()
#model.fit(X_trainpca,Y_train)
#model=MP(n_estimators=10)
model=ET()
model.fit(X_tr,Y_tr)


# In[185]:


model.score(X_train,Y_train)


# In[99]:


#df.head()
ltr=model.predict(X_tr)


# In[100]:


#sm=SM(random_state=42)
#X_ts,Y_ts=sm.fit_sample(X_test,Y_test)


# In[101]:


c=0
for i in Y_test:
    if i==1:
        c+=1
c


# In[186]:


#l=model.predict(X_ts)
model.score(X_test,Y_test)
#model.score(X_ts,Y_ts)
#f1_score(Y_ts,l)


# In[187]:


l1=model.predict(X_test)
f1_score(Y_test,l1)


# In[104]:


Y_test=np.asarray(Y_test).tolist()


# In[178]:


c=0
for i in range(len(l1)):
    if(Y_test[i])==1:
        c+=1
c


# In[189]:


c=0
for i in range(len(l1)):
    if l1[i]!=Y_test[i] and Y_test[i]==1:
        c+=1
print(c)


# In[107]:


#len(Y_test)
#89 right predicted 284 wrong


# In[287]:


c=0
for i in Y_test:
    if i==1:
        c+=1
c


# In[90]:


#X_tr.shape
#dfa["null"]
#dfb["null"].value_counts()


# In[39]:


#plt.scatter(dfa["null"],dfb["null"])


# In[28]:


#df1=df1.drop("null",axis=1)


# In[145]:


#df1[df1["FLAG"]==1].describe()


# In[91]:


#l=Y_test


# In[ ]:


#Y_test----738 out of 8475 are 1.

