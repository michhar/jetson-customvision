# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import sys
import tensorflow as tf
import json
import numpy as np
from PIL import Image
from urllib.request import urlopen
from datetime import datetime
from object_detection import ObjectDetection

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("Runtime error: ", e)

MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'

od_model = None

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()

        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True

        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        with tf.compat.v1.Session(graph=self.graph, config=self.config) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': preprocessed_image[np.newaxis, ...]})
            return outputs[0]

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))
    
def initialize():
    print('Loading model...', end='')
    graph_def = tf.compat.v1.GraphDef()
    with open(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())
    print('Success!')

    print('Loading labels...', end='')
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    print("{} found. Success!".format(len(labels)))
    
    global od_model
    od_model = TFObjectDetection(graph_def, labels)

def predict_url(image_url):
    log_msg("Predicting from url: " + image_url)
    with urlopen(image_url) as image_binary:
        image = Image.open(image_binary)
        return predict_image(image)
    
def predict_image(image):
    log_msg('Predicting image')

    w, h = image.size
    log_msg("Image size: {}x{}".format(w, h))

    predictions = od_model.predict_image(image)

    response = {'id': '',
                'project': '',
                'iteration': '',
                'created': datetime.utcnow().isoformat(),
                'predictions': predictions
                }

    log_msg('Results: ' + json.dumps(response))
    return response
