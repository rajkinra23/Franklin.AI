import os

'''
Config/settings for the object detector.
'''
# Model name
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# Model file path.
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Root to download models.
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
ROOT = 'models/research/object_detection'
PATH_TO_LABELS = os.path.join(ROOT, 'data', 'mscoco_label_map.pbtxt')

# Number of classes for our model.
NUM_CLASSES = 90
