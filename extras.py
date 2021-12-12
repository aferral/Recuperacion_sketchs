# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:55:05 2018

@author: jsaavedr
"""
import numpy as np
import cv2
import tensorflow as tf

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

# --------create image from a set of points ------------------------------------------------------------
def createImage(points):
    x_points = []
    y_points = []
    target_size = 256  # 256
    object_size = 200  # 200
    # reading all points
    for stroke in points:
        x_points = x_points + stroke[0]
        y_points = y_points + stroke[1]
        # min max for each axis
    min_x = min(x_points)
    max_x = max(x_points)
    min_y = min(y_points)
    max_y = max(y_points)

    im_width = np.int(max_x - min_x + 1)
    im_height = np.int(max_y - min_y + 1)

    if im_width > im_height:
        resize_factor = np.true_divide(object_size, im_width)
    else:
        resize_factor = np.true_divide(object_size, im_height)

    t_width = np.int(im_width * resize_factor)
    t_height = np.int(im_height * resize_factor)

    center_x = np.int(sum(x_points) / len(x_points))
    center_y = np.int(sum(y_points) / len(y_points))

    center_x = np.int(t_width * 0.5)
    center_y = np.int(t_height * 0.5)

    t_center_x = np.int(target_size * 0.5)
    t_center_y = np.int(target_size * 0.5)

    offset_x = t_center_x - center_x
    offset_y = t_center_y - center_y

    blank_image = np.zeros((target_size, target_size), np.uint8)
    blank_image[:, :] = 255;
    # cv2.circle(blank_image, (), 1, 1, 8)
    for stroke in points:
        xa = -1
        ya = -1
        for p in zip(stroke[0], stroke[1]):
            x = np.int(
                np.true_divide(p[0] - min_x, im_width) * t_width) + offset_x
            y = np.int(
                np.true_divide(p[1] - min_y, im_height) * t_height) + offset_y
            # if x in range(0,1024) and y in range(0,1024):
            if xa >= 0 and ya >= 0:
                cv2.line(blank_image, (xa, ya), (x, y), 0, 3)
            xa = x
            ya = y
            blank_image[y, x] = 0

    # Resize to 128 x 128
    return cv2.resize(blank_image, (128, 128))
