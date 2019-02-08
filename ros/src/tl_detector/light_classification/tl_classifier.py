from styx_msgs.msg import TrafficLight

import os
from io import BytesIO
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import cv2
import rospy

base_dir = os.path.dirname(os.path.realpath(__file__))

class TLClassifier(object):
    def __init__(self):
	self.sign_classes = ['Red', 'Green', 'Yellow']

	os.chdir(base_dir)

	self.model = load_model('model.h5')
	self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	if image.size <= 0:
	    rospy.loginfo('COLOR: unknown')
            return TrafficLight.UNKNOWN

	img_copy = np.copy(image)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

	img_resize = cv2.resize(img_copy, (32, 32))
	img_resize = np.expand_dims(img_resize, axis=0).astype('float32')

	img_resize = (img_resize / 255.)

	with self.graph.as_default():
	      predict = self.model.predict(img_resize)

	      rospy.loginfo('Prediction: %s', predict)

	      tl_color = self.sign_classes[np.argmax(predict)]

	rospy.loginfo('COLOR: %s', tl_color)
	if tl_color == 'Red':
	      return TrafficLight.RED
	elif tl_color == 'Green':
	      return TrafficLight.GREEN
	elif tl_color == 'Yellow':
	      return TrafficLight.YELLOW
	
        return TrafficLight.UNKNOWN
