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

img_simple=np.load("img_arch1.npy")
img_residual=np.load("img_arch2.npy")

ft_simple = np.load("features_arch1.npy")
ft_residual = np.load("features_arch2.npy")

labels_simple = np.load("labels_arch1.npy")
labels_residual = np.load("labels_arch2.npy")

mapeo = np.load("mapeo_simple_a_res.npy")

dist_simple = metrics.pairwise_distances(ft_simple[0:5000])
dist_residual = metrics.pairwise_distances(ft_residual[0:5000])
to_s = 5
t_near = 5

np.random.seed(45)
chosed = np.random.randint(0,5000,to_s)

for ind,c in enumerate(chosed):
    row=dist_simple[c]
    near = row.argsort()[1:]

    res_index = mapeo[c]
    row_res = dist_residual[res_index]
    near_res = row_res.argsort()[1:]

    save(img_simple[c].reshape(128,128),"query{0}__label_{1}.png".format(ind,labels_simple[c]))
    for j in range(t_near):
        save(img_simple[near[j]].reshape(128,128),"query{0}_{2}_rank{1}.png".format(ind,j,'simple'))
        save(img_residual[near_res[j]].reshape(128,128),"query{0}_{2}_rank{1}.png".format(ind,j,'residual'))
    plt.show()

