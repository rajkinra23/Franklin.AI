'''
This module largely takes the functionality in models/research/object_detection,
and cleans it up on a more re-usable, abstract form.
'''
import object_detection_config as config
import six.moves.urllib as urllib
import numpy as np
import tensorflow as tf
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
import os

'''
Download some pretrained model from the tensorflow repo.

Note that we should probably only have to do this once per model file.
'''
def download_model():
    # Initialize opener.
    opener = urllib.request.URLopener()
    opener.retrieve(os.path.join(config.DOWNLOAD_BASE, config.MODEL_FILE),
                    config.MODEL_FILE)

    # Open the tar file.
    tar_file = tarfile.open(config.MODEL_FILE)

    # Iterate over all the files, to grab the inference graph proto.
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

'''
After downloading the model, the model contents (weights, configs etc) should
be located in a folder with the model name. Specifically, the folder name should
be equivalent to config.MODEL_NAME.
'''
def load_model():
    # Declare the graph.
    graph = tf.Graph()
    with graph.as_default():
        # Initialize a new graph definition.
        graph_def = tf.GraphDef()

        # Load the graph definition from the graph proto file, defined in the
        # config.
        with tf.gfile.GFile(config.PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    # Return back the graph.
    return graph

'''
Utility function to convert an image to an array.
'''
def image_to_array(image):
    w, h = image.size
    arr = image.getdata()
    return np.array(arr).reshape(h, w, 3).astype(np.uint8)

'''
Use the tensorflow object model to run inference for one image.
'''
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors.
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}

            # Prepare map from tensor name to tensor placeholder object.
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)

            # Special processing for detection masks.
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            # Grab the image tensor.
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            # Return the inference.
            return output_dict
