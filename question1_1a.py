
# coding: utf-8

# In[ ]:


import pickle
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans


# In[ ]:


imglist = pickle.load(open("images1.pickle","rb"))
# imgname = pickle.load(open("imagesname1.pickle","rb"))[0]


# In[ ]:


# colorsarr = []
# lis = np.arange(0,256,5)
# #lis = lis.tolist()
# #lis.append(255)
# for i in lis:
#     for j in lis:
#         for k in lis:
#             colorsarr.append([i,j,k])
# colorsarr = np.asarray(colorsarr)
# colorsmodel = KMeans(n_clusters = 64, random_state=42)
# colorsmodel.fit(colorsarr)
# pickle.dump(colorsmodel,open("colorsclusters.pickle","wb"))


# In[ ]:


def autocorr(img, clusters, dist):
    hist = [0]*len(clusters)
    countneigh = [0]*len(clusters)
    count = 0
    for i in range(0,np.shape(img)[0]):
        for j in range(0,np.shape(img)[1]):
            neighbors = neighbor(i,j,np.shape(img)[0], np.shape(img)[1], dist)
            for k in neighbors:
                if(img[i][j] == img[k[0]][k[1]]):
                    hist[img[i][j]]+=1
                    #count+=1
            countneigh[img[i][j]]+=len(neighbors)
    for i in range(len(hist)):
        if(countneigh[i]!=0):
            hist[i] = float(hist[i])/float(countneigh[i])
    return hist

def neighbor(x, y, xmax, ymax, dist):
    points= []
    for i in range(-dist, dist+1):
        for j in range(-dist, dist+1):
            if(i==dist or j==dist or i==(-dist) or j==(-dist)):
                points.append((x+i,y+j))
    p = []
    for i in points:
        if(i[0]>=0 and i[0]<xmax and i[1]>=0 and i[1]<ymax):
            p.append(i)
    return p


# In[ ]:


colorsmodel = pickle.load(open("colorsclusters.pickle","rb"))
clusterlabels = np.unique(colorsmodel.labels_)
dist = [1,3,5,7]


# In[ ]:


corrlist = []
for ll in range(1,6):
    name = "images" + str(ll) + ".pickle"
    imglist = pickle.load(open(name,"rb"))
    length = len(imglist)
    for i in range(length):
        res = []
        img = imglist[i]
        imgnew = cv2.resize(img, (int(np.shape(img)[0]/4), int(np.shape(img)[1]/4) ), interpolation = cv2.INTER_AREA)
        imgnew1 = np.reshape(imgnew,(np.shape(imgnew)[0] * np.shape(imgnew)[1], 3))
        pred = colorsmodel.predict(imgnew1)
        pred = pred.reshape((imgnew.shape[0], imgnew.shape[1]))
        for j in dist:
            result = autocorr(pred, clusterlabels, j)
            res.append(result)
        corrlist.append(res)


# In[ ]:


pickle.dump(corrlist,open("corrnew1.pickle","wb"))


# In[ ]:


corrlist = []
for ll in range(6,11):
    name = "images" + str(ll) + ".pickle"
    imglist = pickle.load(open(name,"rb"))
    length = len(imglist)
    for i in range(length):
        res = []
        img = imglist[i]
        imgnew = cv2.resize(img, (int(np.shape(img)[0]/4), int(np.shape(img)[1]/4) ), interpolation = cv2.INTER_AREA)
        imgnew1 = np.reshape(imgnew,(np.shape(imgnew)[0] * np.shape(imgnew)[1], 3))
        pred = colorsmodel.predict(imgnew1)
        pred = pred.reshape((imgnew.shape[0], imgnew.shape[1]))
        for j in dist:
            result = autocorr(pred, clusterlabels, j)
            res.append(result)
        corrlist.append(res)
        print(i)


# In[ ]:


pickle.dump(corrlist,open("corrnew2.pickle","wb"))


# In[ ]:


corrlist1 = pickle.load(open("corrnew1.pickle","rb"))
corrlist2 = pickle.load(open("corrnew2.pickle","rb"))
kk = corrlist1 + corrlist2
pickle.dump(kk,open("correlogramnew.pickle","wb"))

