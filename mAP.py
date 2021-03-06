import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def fun_car(clas, d_sorted, n):
	elem = clas[d_sorted[n, :]]
	tar = elem[0][0]
	elem = clas[d_sorted[n, 1:]]
	fun = elem==tar
	return fun


def precision(fc):
	return (np.cumsum(fc)/np.arange(1, 1 + len(fc)))[np.nonzero(fc)[0]]


def mean(x):
	return np.mean(x)


def Map(n, clas, d_sorted):
	AP = np.zeros(n)
	for i in range(n):
		funcar = fun_car(clas, d_sorted, i) 	
		pre = precision(funcar)
		pm = mean(pre)
		AP[i] = pm
	t=Map_per_class(AP, clas)
	print(t)
	plt.style.use('ggplot')
	
	plt.title("mAP por clase")
	plt.plot(t)
	plt.savefig("plt.png",dpi=200)

	plt.figure()
	plt.title("Histograma mAP sobre todas las clases")
	plt.hist(t)
	plt.savefig("hist.png",dpi=200)

	return mean(AP)

def MAP(feamat, _class):
	"""
	feamat: (nsamples, ncarac), matriz de caracteristicas.
	_class: (nsamples, 1), vector de clases.
	"""
	ntest, nf = feamat.shape
	d = metrics.pairwise_distances(feamat)
	d_sorted = np.argsort(d, axis=1)
	return Map(ntest, _class, d_sorted)


def Map_per_class(ap, labels):

	labs = np.unique(labels)
	apl = np.zeros(len(labs))
	for i in labs:
		apl[i] = np.mean(ap[np.squeeze(labels==i)])

	return apl

if __name__ == '__main__':
	import numpy as np
	import sys
	
	f = np.load(sys.argv[1])
	labels = np.load(sys.argv[2])
	labels = labels.reshape(labels.shape[0],1)
	map1 = MAP(f,labels)



	print("MAP: {0}".format(map1))
