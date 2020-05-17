
# coding: utf-8

# In[1]:


import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from copy import deepcopy
import timeit
import pywt
from skimage.feature import blob
from scipy.spatial import distance


# In[2]:


def div(arr):
    k = []
    for i in range(4):
        for j in range(4):
            k.append(arr[i*4:i*4+4,j*4:j*4+4])
    return k
def euc(a,b):
    a = np.array(a)
    b = np.array(b)
    c = abs(a - b)
    for i in range(len(c)):
        c[i] = c[i]**2
    return np.sqrt(np.sum(c))
def similarity(des1,des2,t):
    arr = []
    for i in range(len(des1)):
        k = []
        for j in range(len(des2)):
            k.append(distance.euclidean(des1[i],des2[j]))
#             k.append(euc(des1[i],des2[j]))
        k = sorted(k)
        if((k[0]/k[1])<=t):
            arr.append(k[0])
    return len(arr), sum(arr)


# In[3]:


sigmalist = []
for i in range(15):
    sigma1 = (1.24**i)*(1/np.sqrt(2))#http://www.cs.umd.edu/~djacobs/CMSC426/Blob.pdf
    sigmalist.append(sigma1)
kernellist = []
for i in sigmalist:
    k = round(i*6)
    if(k%2==0):
        k = k+1
    kernellist.append(int(k))


def nonmaxsuppr(sigma, img, arr, thresh):
    blobs = []
    boxes = []
    imnew = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
#     feats = []
    for l in range(len(arr)):
        arrl = arr[l]
        for i in range(np.shape(arrl)[0]):
            for j in range(np.shape(arrl)[1]):
                val = arrl[i][j]
                flag = True
                if(val>thresh):
                    for x in range(-2,3):
                        for y in range(-2,3):
                            if((i+x)>=0 and (i+x)<np.shape(arrl)[0] and (j+y)>=0 and (j+y)<np.shape(arrl)[1]):
                                if(arrl[i+x][j+y]>val):
                                    flag = False
                    if(flag):
                        for x in [-1,1]:
                            if((l+x)>=0 and (l+x)<np.shape(arr)[0]):
                                    if(arr[l+x][i][j]>val):
                                        flag = False
                else:
                    flag = False
                if(flag):
                    r = int(2*sigma[l])
                    blobs.append([i,j,r])
    return img, blobs
def descrip(blobs,imnew):
    feats = []
    image = cv2.copyMakeBorder( imnew, 8, 8, 8, 8, cv2.BORDER_CONSTANT)
    for k in blobs:
        i = k[0]
        j = k[1]
        r = k[1]
        if(np.shape(imnew[i-7:i+9,j-7:j+9])[0]==16 and np.shape(imnew[i-7:i+9,j-7:j+9])[1] ==16):
            lis = div(imnew[i-7:i+9,j-7:j+9])
            feat = []
            for kk in range(len(lis)):
                LL, (LH,HL,HH) = pywt.dwt2(lis[kk],'Haar') #LH-dx
                feat.append(np.sum(LH))
                feat.append(np.sum(HL))
                feat.append(np.sum(abs(LH)))
                feat.append(np.sum(abs(HL)))
            feats.append(feat)
        else:
            lis = div(image[i-7+8:i+9+8,j-7+8:j+9+8])
            feat = []
            for kk in range(len(lis)):
                LL, (LH,HL,HH) = pywt.dwt2(lis[kk],'Haar') #LH-dx
                feat.append(np.sum(LH))
                feat.append(np.sum(HL))
                feat.append(np.sum(abs(LH)))
                feat.append(np.sum(abs(HL)))
            feats.append(feat)
    return feats


# In[4]:


imgname = []
for i in range(1,11):
    strr = "imagesname" + str(i) + ".pickle"
    img1 = pickle.load(open(strr,"rb"))
    imgname = imgname + img1


# In[5]:


featlist = pickle.load(open("blobfeatures1.pickle","rb")) + pickle.load(open("blobfeatures2.pickle","rb"))


# In[6]:


querypath = sorted(glob.glob("./train/query/*.txt"))
truthpath = sorted(glob.glob("./train/ground_truth/*.txt"))


# In[8]:


quercorrlist = []
images_path = []
querbloblist = []
start = timeit.default_timer()
for i in range(len(querypath)):#
    f = open(querypath[i],"r").read()
    f = f.replace("oxc1_","")
    f = f.replace('\n',"")
    f = f.split(" ")
    queryimg = cv2.imread(str("./images/") + f[0] + str(".jpg"))
    queryimg = cv2.resize(queryimg, (int(np.shape(queryimg)[1]/4), int(np.shape(queryimg)[0]/4) ), interpolation = cv2.INTER_AREA)
    im1 = deepcopy(queryimg)
    queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2GRAY)
    queryimg = queryimg/255
    queryimg = cv2.GaussianBlur(queryimg,(3,3),0)
    arr = []
    for i in range(len(sigmalist)):
        gau = cv2.GaussianBlur(deepcopy(queryimg),(kernellist[i],kernellist[i]),sigmalist[i])
        conv = cv2.Laplacian(gau,cv2.CV_64F)
        conv = (sigmalist[i]**2)*cv2.Laplacian(gau,cv2.CV_64F)
        arr.append(conv)
    t = sorted(np.reshape(arr,(np.shape(arr)[0]*np.shape(arr)[1]*np.shape(arr)[2])))[::-1][6000]
    im, blobs = nonmaxsuppr(sigmalist, im1, arr, min(t,0.08))#0.1
    blobs = blob._prune_blobs(np.array(blobs),0.6)
    querbloblist.append(blobs)
#     feats = descrip(blobs,im)
#     quercorrlist.append(feats)
#     images_path.append(f[0])
#     for x in blobs:
#         i = x[0]
#         j = x[1]
#         r = x[2]
#         cv2.circle(im, (j, i), r, (0,0,255), 1)
#     strr = "./queryblob/" + f[0] + ".jpg"
#     print(strr)
#     cv2.imwrite(strr,im)
stop = timeit.default_timer()
print((stop-start)/len(querypath))
# pickle.dump(quercorrlist,open("blobdescriptorsquery.pickle","wb"))
# pickle.dump(images_path,open("blobimgnamequery.pickle","wb"))
# pickle.dump(querbloblist,open("blobkeyfeaturesquery.pickle","wb"))


# In[ ]:


# simm = []
# for i in range(len(quercorrlist)):#
#     print(i)
#     query = quercorrlist[i]
#     arr = [0]*len(featlist)
#     for j in range(len(featlist)):
#         arr[j] = similarity(query,featlist[j],0.9)[0]
#     sim = [x for _,x in sorted(zip(arr,imgname))][::-1][:20]
#     simm.append(sim)
#     print(sim)

