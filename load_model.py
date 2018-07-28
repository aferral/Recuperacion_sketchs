
import tensorflow as tf
import numpy as np

import sklearn
from run import get_dataset


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))

    with open("test.html",'w') as f:
        f.write(iframe)
    # display(HTML(iframe))


import os

test_tf_record = "/home/aferral/Escritorio/Recuperacion_sketchs/test.tfrecords"
model_path= "/home/aferral/Escritorio/Recuperacion_sketchs/2018_07_27__11:31:41"
model_path= "/home/inquisidor/Desktop/Recuperacion_sketchs/saved_models_residual_2/train_1/2018_07_27__17:08:51"
ver_grafo = False


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

fc_name = "fc_1/BiasAdd:0"
out_name = "prediction:0"
calc_acc = True
tranf = True


# todo colocar el directorio en '/home/inquisidor/Desktop/Recuperacion_sketchs/train.tfrecords'
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
        pred_batch, real_l, b_vectores, imag  = sess.run([out_name, labels, fc_name, img])
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

