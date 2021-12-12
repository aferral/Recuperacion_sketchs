import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from sklearn import metrics

def save(img,out_name):
    temp = cv2.resize((img*255).astype(np.uint8),(128,128))
    kernel = np.ones((3,3), np.uint8)
    temp = cv2.dilate(temp, kernel, iterations=1)
    temp = (255-temp)
    cv2.imwrite(out_name,temp)

img=np.load(sys.argv[1])
ft = np.load(sys.argv[2])
labels = np.load(sys.argv[3])


dist = metrics.pairwise_distances(ft[0:5000])
to_s = 5
t_near = 5

np.random.seed(3)
chosed = np.random.randint(0,5000,to_s)

for ind,c in enumerate(chosed):
    row=dist[c]
    near = row.argsort()[1:]

    save(img[c].reshape(128,128),"query{0}__label_{1}.png".format(ind,labels[c]))
    for j in range(t_near):
        save(img[near[j]].reshape(128,128),"query{0}_rank{1}.png".format(ind,j))
    plt.show()

