# -*- coding: utf-8 -*-
import tensorflow as tf
from datetime import datetime
import json
import os.path
import subprocess
import random
from extras import createImage
import matplotlib.pyplot as plt


"""
# NO INSTALAR GSUTIL SI SE USA COLAB YA VIENE INSTALADO Y PUEDE TIRAR ERROR AL INTENTAR RE INSTALARLO

Este codigo requiere que este instalado gsutil para instalarlo se puede usar esto (ubuntu)

apt-get install gcc python-dev python-setuptools libffi-dev

apt-get install python-pip

pip uninstall gsutil -y
gsutil version -l


# tambien se requiere pydrive
!pip install -U -q PyDrive

"""


# ----------------------------------------- DEFINICIONES FUNCIONES


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Consigue dataset
def get_dataset(tf_records, epochs, batch_size, shuffle_buffer=500):
    def parse_function(example_proto):
        features = {"image_raw": tf.FixedLenFeature((), tf.string),
                    "label": tf.FixedLenFeature((), tf.int64)
                    }
        parsed_features = tf.parse_single_example(example_proto, features)
        flat_image = tf.decode_raw(parsed_features["image_raw"], tf.uint8)

        reconst = tf.cast(tf.reshape(flat_image, (128, 128)), tf.float32,
                          name='reconstructed_image')

        return reconst, parsed_features["label"]

    dataset = tf.data.TFRecordDataset(tf_records).map(parse_function)
    dataset = dataset.shuffle(shuffle_buffer).repeat(epochs).batch(
        batch_size).cache()

    return dataset



use_upload = False


if use_upload:
    from google.colab import auth
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from oauth2client.client import GoogleCredentials

    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

def save_to_drive(path_file_local,name_remoto,remote_folder_name):
  # Authenticate and create the PyDrive client.
  # This only needs to be done once in a notebook.

  id_f_remota= None

  file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
  for file1 in file_list:
    if file1['title']== remote_folder_name:
      id_f_remota=file1['id']

  # si no esta carpeta fname crearla
  if id_f_remota is None:
    file1 = drive.CreateFile({'title': remote_folder_name,"mimeType": "application/vnd.google-apps.folder"})
    file1.Upload()
    id_f_remota = file1['id']


  # Create & upload a text file.
  uploaded =  drive.CreateFile({'title': name_remoto,'parents' : [{"kind": "drive#fileLink", "id": id_f_remota}]})
  uploaded.SetContentFile(path_file_local)
  uploaded.Upload()
  print('Uploaded file with ID {}'.format(uploaded.get('id')))

  return uploaded.get('id')


# ----------------------------------------- MAIN PREPROCESO



if __name__ == '__main__':


    random.seed(55)
    n=100

    # consigue lista de clases sktech
    a = subprocess.check_output(
        "gsutil ls gs://quickdraw_dataset/full/simplified".split(" "))
    lista = list(filter(lambda x: len(x) != 0, a.decode('utf-8').split('\n')))
    lista_sampled = random.sample(lista, n)
    print(lista_sampled[0:10])



    # Descarga ndjson
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)

        # Descarga elementos en lista
        for sketch_c in lista_sampled:
          cmd=['gsutil', 'cp', '"{0}"'.format(sketch_c), './data']
          subprocess.check_output(cmd)


    # armar train y test
    train_val_points_per_class = 1000
    test_points_per_class = 50
    lista_clases = list(map(lambda x: x.split("/")[-1],lista_sampled))

    images_train=[]
    labels_train=[]
    images_test=[]
    labels_test=[]

    nombres = { i : lista_clases[i] for i in range(len(lista_clases)) }

    for ind,sketch_c in enumerate(lista_clases):
      # load from file-like objects
      if sketch_c.split('.')[-1] == 'ndjson':
        print("Procesando {0}".format(sketch_c))

        current_train = train_val_points_per_class
        current_test = test_points_per_class

        with open(os.path.join("data",sketch_c)) as f:

          for line in f:
            d=json.loads(line)
            img=createImage(d['drawing'])

            if current_test > 0:
              images_test.append(img)
              labels_test.append(ind)
              current_test -= 1
            elif current_train > 0:
              images_train.append(img)
              labels_train.append(ind)
              current_train -= 1
            else:
              break

    # revisar imagenes
    print(len(images_train))
    print(nombres)

    plt.imshow(images_train[0])
    plt.show()



    """# Pasar a tf record"""

    train_filename = 'train.tfrecords'
    test_filename = 'test.tfrecords'



    def to_tf_record(img_list,label_list,out_file_name):
      writer = tf.python_io.TFRecordWriter(out_file_name)

      for i in range(len(img_list)):
        # Escribe en tf record
        img = img_list[i]
        label = label_list[i]
        img_raw = img.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
            'image_raw': _bytes_feature(img_raw),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
      writer.close()

    os.makedirs("tf_records",exist_ok=True)


    # ESCRIBE TF RECORD TRAIN
    path_train=os.path.join("tf_records",train_filename)
    to_tf_record(images_train,labels_train,path_train)


    # Escribe TF record test
    path_test=os.path.join("tf_records",test_filename)
    to_tf_record(images_test,labels_test,path_test)

    # nombre a json
    path_nombres = os.path.join('tf_records','nombres.json')
    with open(path_nombres,'w') as f:
      json.dump(nombres,f)


    """# Subir TF record a drive"""
    if use_upload:
        fecha = datetime.now().strftime("%Y_%m_%d__%H:%M:%S")


        # hacer upload a drive con fecha.
        nombre_remoto = 'train_{0}.tfrecord'.format(fecha)
        save_to_drive(path_train,nombre_remoto,"tarea_deep")

        nombre_remoto = 'test_{0}.tfrecord'.format(fecha)
        save_to_drive(path_test,nombre_remoto,"tarea_deep")


        # escribir nombres tambien
        nombre_remoto = 'nombres_{0}.tfrecord'.format(fecha)
        save_to_drive(path_nombres,nombre_remoto,"tarea_deep")

        # Descarga un archivo desde drive dado ID

        id_train = "14HmQkE1Yv_AwICFCscl0Mnhw6K7JVP0w"
        id_test = "1SRaYRqb50UOXxkVJnF_32QzFj1WzYpUA"
        id_nombres = "15l-51CIHvGK3xFdEgVfxC-ZqSN4hBYKs"



        # recupera train records
        carpeta_destino_local = os.path.join('tf_records','train_recuperado.tfrecord')
        file6 = drive.CreateFile({'id': id_train})
        file6.GetContentFile(carpeta_destino_local)

        # recupera test records
        carpeta_destino_local = os.path.join('tf_records','test_recuperado.tfrecord')
        file6 = drive.CreateFile({'id': id_test})
        file6.GetContentFile(carpeta_destino_local)

        # recupera nombres
        carpeta_destino_local = os.path.join('tf_records','nombres_recuperado.json')
        file6 = drive.CreateFile({'id': id_nombres})
        file6.GetContentFile(carpeta_destino_local)



    """# Revisar el tf record obtenido"""

    # Leer tf record en dataset y comprobar que imagenes sigan bien (y que esten bien mezcladas)
    train_tf_record = os.path.join('tf_records','train.tfrecords')
    try_img = True

    if try_img:

      bs = 20
      dataset=get_dataset(train_tf_record,1,bs,shuffle_buffer=100000)


      # Create iterator
      iterator = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

      with tf.Session() as sess:

        sess.run(iterator.make_initializer(dataset))

        bx,by = iterator.get_next()

        a=sess.run(bx)

        print(by.eval()) # Esto revisa que tan ordenadas estan las clases


        plt.imshow(a[0])
        plt.figure()
        plt.imshow(a[1])
        plt.imshow(a[bs-1])
        plt.show()


