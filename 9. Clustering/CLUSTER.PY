from scipy.cluster.hierarchy import  linkage ,dendrogram
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import os 
os.chdir(r'C:\adv analytics\sir\Cases\Big Mart Sales')
train=pd.read_csv('processed_train.csv')
items_data=train[['Item_Type','i_weight','Item_Visibility','Item_MRP','Item_Outlet_Sales']]
item_mean=items_data.groupby('Item_Type').mean()

scalar=StandardScaler()
i_scaled=scalar.fit_transform(item_mean)
mergings=linkage(i_scaled,method='average')
#use to set  figure size 
fig_size=plt.rcParams['figure.figsize']
fig_size[0]=20
fig_size[1]=10
with plt.style.context("classic"):
            dendrogram(mergings,
            labels=list(item_mean.index),
             leaf_rotation=90,leaf_font_size=10)

plt.show()

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
sil=[]
i_scaled=scalar.fit_transform(item_mean)
for i in np.arange(2,10):
     km=KMeans(n_clusters=i,random_state=2022)
     km.fit(i_scaled)
     labels=km.predict(i_scaled)
     sil.append(silhouette_score(i_scaled,labels))
ks=np.arange(2,10)
i_max=np.argmax(sil)
best_k=ks[i_max]
print("Best k=",best_k)     

