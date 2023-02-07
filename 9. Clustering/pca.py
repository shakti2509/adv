from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
os.chdir(r'C:\adv analytics\Datasets')

milk = pd.read_csv("milk.csv",index_col=0)
scaler=StandardScaler()
scaler.fit(milk)
milkscaled=scaler.transform(milk)

pca=PCA()
prin_comp=pca.fit_transform(milkscaled)
print(milk.shape)
print(prin_comp)

pd_PC=pd.DataFrame(prin_comp,colums=['PC1','PC2','PC3','PC4','PC5'])
pd_PC.var() #or
print(pca.explained_variance_)
#%age variation
print(pca.explained_variance_ratio_)

print(pca.explained_variance_ratio_*100)
###scree plot
ys=np.cumsum(pca.explained_variance_ratio_*100)
xs=np.arange(1,6)
plt.plot(xs,ys)
plt.title("scree plot")
plt.xlabel('principal component')
plt.ylabel('comulative %age variation explained')
plt.show() 


########## biplot#############
from pca import pca
milk = pd.read_csv("milk.csv",index_col=0)
scaler=StandardScaler()
scaler.fit(milk)
milkscaled=scaler.transform(milk)
milkscaled=pd.DataFrame(milkscaled,columns=milk.columns,index=milk.index)

model=pca()
results=model.fit_transform(milkscaled)
fig,ax=model.biplot(label=True,legend=False)

#################### iris data set #############
import seaborn as sns 
iris=pd.read_csv('iris.csv',index_col=4)
scaler=StandardScaler()
scaler.fit(iris)
irisscaled=scaler.transform(iris)

pca=PCA()
prin_comp=pca.fit_transform(irisscaled)
print(iris.shape)
print(prin_comp)

pd_PC=pd.DataFrame(prin_comp,columns=['PC1','PC2','PC3','PC4'])
pd_PC.var() #or
print(pca.explained_variance_)
#%age variation
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_*100)

sns.scatterplot(data=iris,x='Sepal.Length',y='Sepal.Width',hue='Species') 

plt.xlabel('sepal length')
plt.ylabel('sepal weidth')
plt.show()


iris=pd.read_csv('iris.csv')

pd_PC=pd.DataFrame(prin_comp,columns=['PC1','PC2','PC3','PC4'])
pd_PC['Species']=iris['Species']
sns.scatterplot(data=pd_PC,x='PC1',y='PC2',hue='Species')
plt.title('scatter plot of PC1 and PC2 data ')
plt.show()


############ wine data set


wine=pd.read_csv('wine.csv',index_col=0)
scaler=StandardScaler()
scaler.fit(wine)
winescaled=scaler.transform(wine)

pca=PCA()
prin_comp=pca.fit_transform(winescaled)
print(wine.shape)
print(prin_comp)

pd_PC=pd.DataFrame(prin_comp,columns=['PC1','PC2','PC3','PC4',
                                      'PC5','PC6','PC7','PC8',
                                      'PC9','PC10','PC11','PC12',
                                      'PC13'])


#instead of puting all pcs values in columns or or or 
pd_PC=pd.DataFrame(prin_comp,columns=['PC'+str(i) for i in np.arange(1,14)])
pd_PC.var() #or
print(pca.explained_variance_)
#%age variation
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_*100)


ys=np.cumsum(pca.explained_variance_ratio_*100)


wine=pd.read_csv('wine.csv')

pd_PC=pd.DataFrame(prin_comp,columns=['PC'+str(i) for i in np.arange(1,14)])
pd_PC['Class']=wine['Class']
sns.scatterplot(data=pd_PC,x='PC1',y='PC2',hue='Class')
plt.title('scatter plot of PC1 and PC2 data ')
plt.show()


###########PCA###########
train=pd.read_csv(r'C:\adv analytics\sir\Cases\Big Mart Sales\processed_train.csv')
X=train.drop(['Item_Identifier','Item_Outlet_Sales','Outlet_Identifier'],axis=1)
dum_X=pd.get_dummies(X,drop_first=True)

scaler=StandardScaler()
scaler.fit(dum_X)
trainscaled=scaler.transform(dum_X)

pca=PCA()
prin_comp=pca.fit_transform(trainscaled)
print(dum_X.shape)
print(prin_comp)

#pd_PC=pd.DataFrame(prin_comp,columns=['PC1','PC2','PC3','PC4',
 #                                     'PC5','PC6','PC7','PC8',
#                                      'PC9','PC10','PC11','PC12',
#                                      'PC13'])


#instead of puting all pcs values in columns or or or 
pd_PC=pd.DataFrame(prin_comp,columns=['PC'+str(i) for i in np.arange(1,28)])
pd_PC.var() #or
print(pca.explained_variance_)
#%age variation
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_*100)


ys=np.cumsum(pca.explained_variance_ratio_*100)


train=pd.read_csv('train.csv')

#pd_PC=pd.DataFrame(prin_comp,columns=['PC'+str(i) for i in np.arange(1,28)])
xs=np.arange(1,28)
plt.plot(xs,ys)
plt.title("scree plot")
plt.xlabel('principal component')
plt.ylabel('comulative %age variation explained')
plt.show() 


#############################protein.csv########################
Protein=pd.read_csv('Protein.csv',index_col=0)
scaler=StandardScaler()
scaler.fit(Protein)
Proteinscaled=scaler.transform(Protein)

pca=PCA()
prin_comp=pca.fit_transform(Proteinscaled)
print(Protein.shape)
print(prin_comp)

pd_PC=pd.DataFrame(prin_comp,columns=['PC'+str(i) for i in np.arange(1,10)])
pd_PC.var() #or
print(pca.explained_variance_)
#%age variation
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_*100)


ys=np.cumsum(pca.explained_variance_ratio_*100)
train=pd.read_csv('Protein.csv')
xs=np.arange(1,10)
plt.plot(xs,ys)
plt.title("scree plot")
plt.xlabel('principal component')
plt.ylabel('comulative %age variation explained')
plt.show() 


#Biplot:-
from pca import pca
milk = pd.read_csv("Protein.csv",index_col=0)
scaler=StandardScaler()
scaler.fit(Protein)
Proteinscaled=scaler.transform(Protein)
Proteinscaled=pd.DataFrame(Proteinscaled,columns=Protein.columns,index=Protein.index)
#what we are hiding we get by index=protein.index
model=pca()
results=model.fit_transform(Proteinscaled)
fig,ax=model.biplot(label=True,legend=False)


#sil:
from sklearn.metrics import silhouette_score
sil = []
for i in np.arange(2,10):
    km = KMeans(n_clusters=i, random_state=2022)
    km.fit(Proteinscaled)
    labels = km.predict(Proteinscaled)
    sil.append(silhouette_score(Proteinscaled, labels))

Ks = np.arange(2,10)
i_max = np.argmax(sil)
best_k = Ks[i_max]
print("Best K =", best_k)

#Generrating labels
km = KMeans(n_clusters=best_k, random_state=2022)
km.fit(Proteinscaled)
labels = km.predict(Proteinscaled)

#scatter plot of 3 clusters:-
pd_PC=pd.DataFrame(prin_comp,columns=['PC'+str(i) for i in np.arange(1,10)])
pd_PC['Cluster']=labels

pd_PC['Cluster']=pd_PC['Cluster'].astype('category')
sns.scatterplot(data=pd_PC,x='PC1',y='PC2',hue='Cluster')
plt.title('scatter plot of PC1 and PC2 data ')
plt.show()
