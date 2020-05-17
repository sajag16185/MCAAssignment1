
# coding: utf-8

# In[6]:


import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import timeit
from skimage.feature import blob


# In[7]:


def div(arr):
    k = []
    for i in range(4):
        for j in range(4):
            k.append(arr[i*4:i*4+4,j*4:j*4+4])
    return k


# In[8]:


# from scipy import ndimage
    
# def l_o_g(x,y,sigma):
#     k = (x**2 + y**2)/(2*(sigma**2))
#     m = (1 - k)* np.exp(-k) / (np.pi*(sigma**2))
#     return m
# def createlogmat(size, sigma):
#     mat = np.zeros((size,size))
#     m = int(size/2)
#     for x in range(size):
#         for y in range(size):
#             mat[x][y] = l_o_g(x - m, y-m, sigma)
#     return mat

arr = []

sigmalist = []
for i in range(15):
    sigma1 = (1.24**i)*(1/np.sqrt(2))
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
    for l in range(len(arr)):
        arrl = arr[l]
        for i in range(np.shape(arrl)[0]):
            for j in range(np.shape(arrl)[1]):
                val = arrl[i][j]
                flag = True
                if(val>thresh):
                    for x in range(-1,2):
                        for y in range(-1,2):
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


# In[9]:


print(sigmalist)
print(kernellist)


# In[10]:


#featlist = []
keypointlist = []


# In[11]:


image_path = glob.glob("./images/*.jpg")
images_path = []
start = timeit.default_timer()
for mm in image_path:
    print(mm)
    img = cv2.imread(mm)
    images_path.append(mm)
    im = cv2.resize(img, (int(np.shape(img)[1]/4), int(np.shape(img)[0]/4) ), interpolation = cv2.INTER_AREA)
    im1 = deepcopy(im)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img = img/255
    arr = []
    img = cv2.GaussianBlur(img,(3,3),0)
    for i in range(len(sigmalist)):
        gau = cv2.GaussianBlur(deepcopy(img),(kernellist[i],kernellist[i]),sigmalist[i])
        conv = cv2.Laplacian(gau,cv2.CV_64F)
        conv = (sigmalist[i]**2)*cv2.Laplacian(gau,cv2.CV_64F)
        arr.append(conv)
    t = sorted(np.reshape(arr,(np.shape(arr)[0]*np.shape(arr)[1]*np.shape(arr)[2])))[::-1][6000]
    im, blobs = nonmaxsuppr(sigmalist, im1, arr, min(t,0.08))
    blobs = blob._prune_blobs(np.array(blobs),0.6)
    keypointlist.append(blobs)
stop = timeit.default_timer()
print((stop-start)/len(keypointlist))
pickle.dump(keypointlist,open("blobfeaturesall.pickle","wb"))
pickle.dump(images_path,open("blobimgname.pickle","wb"))

