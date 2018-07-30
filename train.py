
import tensorflow as tf
from arquitecturas import redResidua,redSimple
from preproceso import get_dataset
import os


if __name__ == "__main__":

    # Parametros iniciales
    train_tf_record = "tf_records/train.tfrecords"
    total=100000
    epochs=20
    nombre_modelo = 'residual'

    # definir dataset
    dataset=get_dataset(train_tf_record,epochs,100,shuffle_buffer=40000)


    # construir arquitectura
    iterator = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
    imagenes,labels = iterator.get_next()
    out = redResidua(imagenes)
    pred = tf.nn.softmax(out, name='prediction')
    loss = tf.losses.sparse_softmax_cross_entropy(labels, out)
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    # cosas para metricas
    prediction = tf.argmax(out, 1)
    equality = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))


    with tf.Session() as sess:

        # inicializar
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(iterator.make_initializer(dataset))
        saver = tf.train.Saver()

        it_per_epoch = total // 200

        # train loop
        for epoch in range(epochs):
            for iteracion in range(it_per_epoch):
                if iteracion % 30 == 0:
                    l, _, acc = sess.run([loss, train_step, accuracy])
                    print("Iteracion {0} loss {1} accuracy {2}".format(iteracion, l, acc))
                else:
                    sess.run(train_step)

                if (iteracion % 100 == 0) or (iteracion == (it_per_epoch-1)):
                    path_model_checkpoint = os.path.join('models', nombre_modelo)
                    os.makedirs(path_model_checkpoint, exist_ok=True)
                    path_checkpoint = os.path.join(path_model_checkpoint, 'modelo')
                    saver.save(sess, path_checkpoint)
            print("Fin epoch {0}".format(epoch))


