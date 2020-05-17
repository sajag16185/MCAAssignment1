
# coding: utf-8

# In[19]:


import pickle
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
import timeit
import matplotlib.pyplot as plt


# In[2]:


colorsmodel = pickle.load(open("colorsclusters.pickle","rb"))
corrlist = pickle.load(open("correlogramnew.pickle","rb"))


# In[3]:


clustercentres = colorsmodel.cluster_centers_
clusterlabels = np.unique(colorsmodel.labels_)


# In[4]:


def sim1(hist1,hist2):
    simmat = [0]*len(hist1)
    for i in range(len(hist1)):
        simmat[i] = abs(hist1[i] - hist2[i])/(hist1[i] + hist2[i] + 1)
    return sum(simmat)/len(hist1)

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
                    count+=1
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


# In[5]:


querypath = glob.glob("./train/query/*.txt")
querypath = sorted(querypath)
truthpath = sorted(glob.glob("./train/ground_truth/*.txt"))
truthpath = sorted(truthpath)


# In[42]:


queryname = []
for i in range(len(querypath)):
    f = open(querypath[i],"r").read()
    f = f.replace("oxc1_","")
    f = f.replace('\n',"")
    f = f.split(" ")
    queryname.append(f[0])


# In[6]:


#sim1
dist = [1,3,5,7]
quercorrlist = []
start = timeit.default_timer()
for i in range(len(querypath)):
    f = open(querypath[i],"r").read()
    f = f.replace("oxc1_","")
    f = f.replace('\n',"")
    f = f.split(" ")
    #img = cv2.imread(str("./images/") + f[0] + str(".jpg"))
    #queryimg = img[int(float(f[1])) : int(float(f[3])), int(float(f[2])): int(float(f[4])), :]
    queryimg = cv2.imread(str("./images/") + f[0] + str(".jpg"))
    res = []
    queryimg = cv2.resize(queryimg, (int(np.shape(queryimg)[0]/4), int(np.shape(queryimg)[1]/4) ), interpolation = cv2.INTER_AREA)
    imgnew1 = np.reshape(queryimg,(np.shape(queryimg)[0] * np.shape(queryimg)[1], 3))
    pred = colorsmodel.predict(imgnew1)
    pred = pred.reshape((queryimg.shape[0], queryimg.shape[1]))
    for j in dist:
        result = autocorr(pred, clusterlabels, j)
        res.append(result)
    print(i)
    quercorrlist.append(res)
stop = timeit.default_timer()
print("Total Time = ", stop - start)
print("Average Time = ", (stop - start)/len(querypath))

# stop = timeit.default_timer()
# print("Total Time = ", stop - start)
# print("Average Time = ", (stop - start)/len(querypath))


# In[9]:


pickle.dump(quercorrlist,open("quercorr.pickle","wb"))


# In[7]:


start = timeit.default_timer()
image_path = glob.glob("./images/*.jpg")
simval = np.zeros((len(quercorrlist),len(corrlist)))
for i in range(len(quercorrlist)):
    for j in range(len(corrlist)):
        for k in range(4):
            simval[i][j] += sim1(quercorrlist[i][k], corrlist[j][k])
stop = timeit.default_timer()
print("Total Time = ", stop - start)
print("Average Time = ", (stop - start)/len(querypath))


# In[31]:


count = 0
simm = simval
neigh = 10
precision_g = [0]*len(simm)
precision_j = [0]*len(simm)
precision_o = [0]*len(simm)
precision_t = [0]*len(simm)
recall_g = [0]*len(simm)
recall_j = [0]*len(simm)
recall_o = [0]*len(simm)
recall_t = [0]*len(simm)
f1score_g = [0]*len(simm)
f1score_j = [0]*len(simm)
f1score_o = [0]*len(simm)
f1score_t = [0]*len(simm)
good = [0]*len(simm)
junk = [0]*len(simm)
ok = [0]*len(simm)
for i in range(len(simm)):
    lis = sorted(simm[i])#[::-1]
    imgs = []
    f = open(truthpath[count], "r").read()
    f = f.split('\n')
    f1 = open(truthpath[count+1], "r").read()
    f1 = f1.split('\n')
    f2 = open(truthpath[count+2], "r").read()
    f2 = f2.split('\n')
    for j in range(neigh):
        l = image_path[np.where(simm[i]==lis[j])[0][0]].split("\\")[-1]
        l = l.replace(".jpg","")
        imgs.append(l)
    for k in imgs:
        if k in f:
            good[i]+=1
        if k in f1:
            junk[i]+=1
        if k in f2:
            ok[i]+=1
    count+=3
    precision_g[i] = good[i]/(20)
    precision_j[i] = junk[i]/(20)
    precision_o[i] = ok[i]/(20)
    precision_t[i] = (good[i] + junk[i] + ok[i])/(20)
    recall_g[i] = good[i]/len(f)
    recall_j[i] = junk[i]/len(f1)
    recall_o[i] = ok[i]/len(f2)
    recall_t[i] = (good[i] + junk[i] + ok[i])/(len(f) + len(f1) + len(f2))
    if((precision_g[i] + recall_g[i]) != 0):
        f1score_g[i] = (2*precision_g[i]*recall_g[i])/(precision_g[i] + recall_g[i])
    if((precision_j[i] + recall_j[i]) != 0):
        f1score_j[i] = (2*precision_j[i]*recall_j[i])/(precision_j[i] + recall_j[i])
    if((precision_o[i] + recall_o[i]) != 0):
        f1score_o[i] = (2*precision_o[i]*recall_o[i])/(precision_o[i] + recall_o[i])
    if((precision_t[i] + recall_t[i]) != 0):
        f1score_t[i] = (2*precision_t[i]*recall_t[i])/(precision_t[i] + recall_t[i])
print("Total good retrievals = ",sum(good))
print("Total ok retrievals = ",sum(ok))
print("Total junk retrievals = ",sum(junk))
print("Average good retrievals = ",sum(good)/len(simm))
print("Average ok retrievals = ",sum(ok)/len(simm))
print("Average junk retrievals = ",sum(junk)/len(simm))
print("Total retrievals = ", sum(good)+ sum(ok) + sum(junk))
print("Average retrievals = ", (sum(good)+ sum(ok) + sum(junk))/len(simm))
print("Maximum Precision overall = ",max(precision_t))
print("Minimum Precision overall = ",min(precision_t))
print("Average Precision overall = ",sum(precision_t)/len(precision_t))
print("Maximum recall overall = ",max(recall_t))
print("Minimum recall overall = ",min(recall_t))
print("Average recall overall = ",sum(recall_t)/len(recall_t))
print("Maximum f1 score overall = ",max(f1score_t))
print("Minimum f1 score overall = ",min(f1score_t))
print("Average f1 score overall = ",sum(f1score_t)/len(f1score_t))


# In[48]:


reqvalues =[[] for _ in range(13)]
kn = 31
for neigh in range(1,kn):
    count = 0
    simm = simval
    precision_g = [0]*len(simm)
    precision_j = [0]*len(simm)
    precision_o = [0]*len(simm)
    precision_t = [0]*len(simm)
    recall_g = [0]*len(simm)
    recall_j = [0]*len(simm)
    recall_o = [0]*len(simm)
    recall_t = [0]*len(simm)
    f1score_g = [0]*len(simm)
    f1score_j = [0]*len(simm)
    f1score_o = [0]*len(simm)
    f1score_t = [0]*len(simm)
    good = [0]*len(simm)
    junk = [0]*len(simm)
    ok = [0]*len(simm)
    for i in range(len(simm)):
        lis = sorted(simm[i])#[::-1]
        imgs = []
        f = open(truthpath[count], "r").read()
        f = f.split('\n')
        f1 = open(truthpath[count+1], "r").read()
        f1 = f1.split('\n')
        f2 = open(truthpath[count+2], "r").read()
        f2 = f2.split('\n')
        for j in range(neigh):
            l = image_path[np.where(simm[i]==lis[j])[0][0]].split("\\")[-1]
            l = l.replace(".jpg","")
            imgs.append(l)
        for k in imgs:
            if k in f:
                good[i]+=1
            if k in f1:
                junk[i]+=1
            if k in f2:
                ok[i]+=1
        count+=3
        precision_g[i] = good[i]/(neigh)
        precision_j[i] = junk[i]/(neigh)
        precision_o[i] = ok[i]/(neigh)
        precision_t[i] = (good[i] + junk[i] + ok[i])/(20)
        recall_g[i] = good[i]/len(f)
        recall_j[i] = junk[i]/len(f1)
        recall_o[i] = ok[i]/len(f2)
        recall_t[i] = (good[i] + junk[i] + ok[i])/(len(f) + len(f1) + len(f2))
        if((precision_g[i] + recall_g[i]) != 0):
            f1score_g[i] = (2*precision_g[i]*recall_g[i])/(precision_g[i] + recall_g[i])
        if((precision_j[i] + recall_j[i]) != 0):
            f1score_j[i] = (2*precision_j[i]*recall_j[i])/(precision_j[i] + recall_j[i])
        if((precision_o[i] + recall_o[i]) != 0):
            f1score_o[i] = (2*precision_o[i]*recall_o[i])/(precision_o[i] + recall_o[i])
        if((precision_t[i] + recall_t[i]) != 0):
            f1score_t[i] = (2*precision_t[i]*recall_t[i])/(precision_t[i] + recall_t[i])
    
    reqvalues[0].append(sum(good)/len(simm))#average good queries
    reqvalues[1].append(sum(ok)/len(simm))#average ok queries
    reqvalues[2].append(sum(junk)/len(simm))#Average junk queries
    reqvalues[3].append((sum(good)+ sum(ok) + sum(junk))/len(simm))#Average retrievals
    reqvalues[4].append(max(precision_t))
#     print(precision_t.index(max(precision_t)))
#     print(queryname[precision_t.index(max(precision_t))])
    reqvalues[5].append(min(precision_t))
#     print(precision_t.index(min(precision_t)))
#     print(queryname[precision_t.index(min(precision_t))])
    reqvalues[6].append(sum(precision_t)/len(precision_t))
    reqvalues[7].append(max(recall_t))
#     print(recall_t.index(max(recall_t)))
#     print(queryname[recall_t.index(max(recall_t))])
    reqvalues[8].append(min(recall_t))
#     print(recall_t.index(min(recall_t)))
#     print(queryname[recall_t.index(min(recall_t))])
    reqvalues[9].append(sum(recall_t)/len(recall_t))
    reqvalues[10].append(max(f1score_t))
#     print(f1score_t.index(max(f1score_t)))
#     print(queryname[f1score_t.index(max(f1score_t))])
    reqvalues[11].append(min(f1score_t))
    print(f1score_t.index(min(f1score_t)))
    print(queryname[f1score_t.index(min(f1score_t))])
    reqvalues[12].append(sum(f1score_t)/len(f1score_t))


# In[33]:


x = np.arange(1,kn)
plt.plot(x, reqvalues[0], label = 'Average good queries retrieved')
plt.plot(x, reqvalues[1], label = 'Average ok queries retrieved')
plt.plot(x, reqvalues[2], label = 'Average junk queries retrieved')
plt.plot(x, reqvalues[3], label = 'Average total queries retrieved')
plt.legend(loc='best')
plt.ylabel('Total number of images retrieved/Number of queries')
plt.xlabel('Number of images retrieved')


# In[34]:


plt.plot(x, reqvalues[4], label = 'Maximum Precision')
plt.plot(x, reqvalues[5], label = 'Minimum Precision')
plt.plot(x, reqvalues[6], label = 'Average Precision')
plt.legend(loc='best')
plt.ylabel('Precision')
plt.xlabel('Number of images retrieved')


# In[35]:


plt.plot(x, reqvalues[7], label = 'Maximum Recall')
plt.plot(x, reqvalues[8], label = 'Minimum Recall')
plt.plot(x, reqvalues[9], label = 'Average Recall')
plt.legend(loc='best')
plt.ylabel('Recall')
plt.xlabel('Number of images retrieved')


# In[36]:


plt.plot(x, reqvalues[10], label = 'Maximum F1 score')
plt.plot(x, reqvalues[11], label = 'Minimum F1 score')
plt.plot(x, reqvalues[12], label = 'Average F1 score')
plt.legend(loc='best')
plt.ylabel('F1 score')
plt.xlabel('Number of images retrieved')


# In[21]:


# print("Maximum Precision good = ",max(precision_g))
# print("Minimum Precision good = ",min(precision_g))
# print("Average Precision good = ",sum(precision_g)/len(precision_g))
# print("Maximum Precision ok = ",max(precision_o))
# print("Minimum Precision ok = ",min(precision_o))
# print("Average Precision ok = ",sum(precision_o)/len(precision_o))
# print("Maximum Precision junk = ",max(precision_j))
# print("Minimum Precision junk = ",min(precision_j))
# print("Average Precision junk = ",sum(precision_j)/len(precision_j))


# In[12]:


# print("Maximum recall good = ",max(recall_g))
# print("Minimum recall good = ",min(recall_g))
# print("Average recall good = ",sum(recall_g)/len(recall_g))
# print("Maximum recall ok = ",max(recall_o))
# print("Minimum recall ok = ",min(recall_o))
# print("Average recall ok = ",sum(recall_o)/len(recall_o))
# print("Maximum recall junk = ",max(recall_j))
# print("Minimum recall junk = ",min(recall_j))
# print("Average recall junk = ",sum(recall_j)/len(recall_j))


# In[13]:


# print("Maximum f1 score good = ",max(f1score_g))
# print("Minimum f1 score good = ",min(f1score_g))
# print("Average f1 score good = ",sum(f1score_g)/len(f1score_g))
# print("Maximum f1 score ok = ",max(f1score_o))
# print("Minimum f1 score ok = ",min(f1score_o))
# print("Average f1 score ok = ",sum(f1score_o)/len(f1score_o))
# print("Maximum f1 score junk = ",max(f1score_j))
# print("Minimum f1 score junk = ",min(f1score_j))
# print("Average f1 score junk = ",sum(f1score_j)/len(f1score_j))


# In[38]:




