
# coding: utf-8

# In[1]:


import pickle
import glob
import cv2
import numpy as np
from copy import deepcopy
from skimage.feature import hessian_matrix_det, blob# blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import timeit


# In[2]:


img = pickle.load(open("images1.pickle","rb"))


# In[3]:


sigmalist = []
for k in range(1,10):
    sigmalist.append(1.2*k)
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


# In[4]:


start = timeit.default_timer()
keypointlist = []
images_path = []
for mm in range(1,11):
    strk = "images" + str(mm) + ".pickle"
    strm = "imagesname" + str(mm) + ".pickle"
    img = pickle.load(open(strk,"rb"))
    imgname = pickle.load(open(strm,"rb"))
    for ind in range(len(imgname)):
        print(imgname[ind])
        im = cv2.resize(img[ind], (int(np.shape(img[ind])[1]/4), int(np.shape(img[ind])[0]/4) ), interpolation = cv2.INTER_AREA)
        im1 = deepcopy(im)
        img[ind] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img[ind] = img[ind]/255
        arr = []
        for i in range(len(sigmalist)):
            conv = hessian_matrix_det(img[ind], int(sigmalist[i]))
            arr.append(conv)
        t = sorted(np.reshape(arr,(np.shape(arr)[0]*np.shape(arr)[1]*np.shape(arr)[2])))[::-1][6000]
        im, blobs = nonmaxsuppr(sigmalist, im1, arr, t)
        blobs = blob._prune_blobs(np.array(blobs),0.6)
        keypointlist.append(blobs)
        images_path.append(imgname[ind])
        for x in blobs:
            i = x[0]
            j = x[1]
            r = x[2]
            cv2.circle(im, (j, i), r, (0,0,255), 1)
        strr = "./surfblob/" + imgname[ind].split("\\")[1]
        cv2.imwrite(strr,im)
stop = timeit.default_timer()
print((stop - start)/len(keypointlist))
pickle.dump(keypointlist,open("./Q3/surf_keypts_all.pickle","wb"))
pickle.dump(images_path,open("./Q3/surf_all_name.pickle","wb"))


# In[5]:


querypath = sorted(glob.glob("./train/query/*.txt"))
querkeypointslist = []
querynames = []
start = timeit.default_timer()
length = len(querypath)
for ind in range(length):#
    f = open(querypath[ind],"r").read()
    f = f.replace("oxc1_","")
    f = f.replace('\n',"")
    f = f.split(" ")
    print(f)
    queryimg = cv2.imread(str("./images/") + f[0] + str(".jpg"))
    queryimg = cv2.resize(queryimg, (int(np.shape(queryimg)[1]/4), int(np.shape(queryimg)[0]/4) ), interpolation = cv2.INTER_AREA)
    im1 = deepcopy(queryimg)
    queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2GRAY)
    queryimg = queryimg/255
    arr = []
    for i in range(len(sigmalist)):
        conv = hessian_matrix_det(queryimg, int(sigmalist[i]))
        arr.append(conv)
    t = sorted(np.reshape(arr,(np.shape(arr)[0]*np.shape(arr)[1]*np.shape(arr)[2])))[::-1][6000]
    print(t)
    im, blobs = nonmaxsuppr(sigmalist, im1, arr,t)# min(t,0.003)
    blobs = blob._prune_blobs(np.array(blobs),0.6)
    querkeypointslist.append(blobs)
    querynames.append(f[0])
    for x in blobs:
        i = x[0]
        j = x[1]
        r = x[2]
        cv2.circle(im, (j, i), r, (0,0,255), 1)
    strr = "./surfqueryblob/" + f[0] + ".jpg"
    cv2.imwrite(strr,im)
stop = timeit.default_timer()
print(stop-start)
pickle.dump(querkeypointslist,open("./Q3/surf_kpt_query.pickle","wb"))
pickle.dump(querynames,open("./Q3/surf_query_name.pickle","wb"))

