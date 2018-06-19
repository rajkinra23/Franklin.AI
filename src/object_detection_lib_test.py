'''
Unit test to test object detection.
'''
import object_detection_lib as odl
import object_detection_config as config
import os
from PIL import Image
from matplotlib import pyplot as plt
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

# Constants for testing.
IMAGE_ROOT = 'screens'
TEST_IMAGE = 1

'''
Get the test images from the root, and convert them to image arrays.
'''
def get_test_images():
    # Create container for test image arrays. Each element should be
    # a numpy array.
    imgs = []

    # Iterate through each image in the IMAGE_ROOT:
    # (1) Open with Pillow's Image object.
    # (2) Use the utility to convert to a numpy array
    # (3) append to the container.
    for test_image in os.listdir(IMAGE_ROOT):
        if test_image != '.keep.txt':
            image_path = os.path.join(IMAGE_ROOT, test_image)
            pil_image = Image.open(image_path)
            image_np_array = odl.image_to_array(pil_image)
            imgs.append(image_np_array)

    # Return the images.
    return imgs

'''
Test running inference on a single image.
'''
def test_object_detection():
    # Create the label/category maps for drawing the bbox objects.
    label_map = label_map_util.load_labelmap(config.PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=config.NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the model.
    model = odl.load_model()

    # Get test images.
    test_image = get_test_images()[TEST_IMAGE]

    # Feed the model and the first images to the inference runner.
    output_dict = odl.run_inference_for_single_image(test_image, model)

    # Log the detections.
    objs = set()
    scores = output_dict['detection_scores']
    classes = output_dict['detection_classes']
    for i in range(len(scores)):
        score = scores[i]
        c = classes[i]
        if score >= 0.2:
            objs.add(category_index[c]['name'])
    print("Objects detected: %s" % str(objs))

    # Save the output.
    vis_util.visualize_boxes_and_labels_on_image_array(
        test_image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.4)
    vis_util.save_image_array_as_png(test_image, 'test_screens/test_image.png')

if __name__ == '__main__':
    test_object_detection()
