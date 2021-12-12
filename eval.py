import tensorflow as tf
import numpy as np
import sklearn
import os
from extras import show_graph

test_tf_record = "/home/aferral/Escritorio/Recuperacion_sketchs/test.tfrecords"
model_path= "./models/residual"
ver_grafo = False

"""
Como usar esto???

- Para recuperar las features de un arquitectura es necesario tener el nombre del tensor. Para obtener este nombre se
observa el grafo con la funcion en ver_grafo

Se anotan los tensores en fully_conected, out_name. IMPORTANTE los tensores se indican con :0 ya que la operacion es 
"acaNombre" y sus salidas son :x  :0 quiere decir la primera salida (usualmente tienen solo 1 salida)

"""

# Ver grafo
if ver_grafo:
	with tf.Session() as sess:
		archivos=os.listdir(model_path)
		meta_f = list(filter(lambda x: x.split('.')[-1] == 'meta',archivos   ))[0]
		f_p_meta = os.path.join(model_path,meta_f)
		new_saver = tf.train.import_meta_graph(f_p_meta)
		new_saver.restore(sess, tf.train.latest_checkpoint(model_path))

		graph = tf.get_default_graph()
		show_graph(graph)


fully_conected = "fully_connected/BiasAdd:0"
out_name = "prediction:0"


# calcular accuracy en test
with tf.Session() as sess:
    archivos=os.listdir(model_path)
    meta_f = list(filter(lambda x: x.split('.')[-1] == 'meta',archivos   ))[0]
    f_p_meta = os.path.join(model_path,meta_f)
    new_saver = tf.train.import_meta_graph(f_p_meta)
    new_saver.restore(sess, tf.train.latest_checkpoint(model_path))

    predicted_test_set = []
    real_test_set = []
    features = []
    images = []

    # crear dataset
    graph = sess.graph
    make_init = graph.get_operation_by_name('make_initializer')
    sess.run(make_init)

    labels = graph.get_operation_by_name("IteratorGetNext").values()[1]


    cant_batch = 5000 // 100 #200

    img = graph.get_operation_by_name("IteratorGetNext").values()[0]

    labels = graph.get_operation_by_name("IteratorGetNext").values()[1]


    for i in range(cant_batch):
        print("iteracion {0}".format(i))

        # calc predicted
        pred_batch, real_l, b_vectores, imag  = sess.run([out_name, labels, fully_conected, img])
        pred_batch = pred_batch.argmax(axis=1)

        for elem in real_l:
            real_test_set.append(elem)
        for elem in pred_batch:
            predicted_test_set.append(elem)

        for elem in b_vectores:
            features.append(elem)
        for elem in imag:
            images.append(elem.flatten())

    features = np.vstack(features)
    images = np.vstack(images)

    res=sklearn.metrics.accuracy_score(real_test_set, predicted_test_set)
    print("La accuracy fue {0}".format(res))

    np.save('features.npy', features)
    np.save("labels.npy", real_test_set)
    np.save('imagenes.npy', images)
    # imagenes de las features []

