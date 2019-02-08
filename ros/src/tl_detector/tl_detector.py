#!/usr/bin/env python
import rospy
import os
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import math
import PyKDL

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
	self.waypoints_2d = None
	self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        ### for save image  ####
        self.has_image = False
        self.picture_num = 0
        ############################

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
	self.is_site = self.config['is_site']

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
	if not self.waypoints_2d:
	    self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
	    self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

	rospy.loginfo('image_cb running COLOR: %s', state)
	rospy.loginfo('image_cb running COLOR self.state: %s', self.state)
	rospy.loginfo('image_cb running COLOR self.state_count: %s', self.state_count)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
	    rospy.loginfo('red published %s', light_wp)
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.waypoint_tree.query([x,y], 1)[1]

    def crop_image(self, closest_light):
        cv_img = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")      
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link", "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link", "/world", now)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        traffic_x = closest_light.pose.pose.position.x
        traffic_y = closest_light.pose.pose.position.y
        traffic_z = closest_light.pose.pose.position.z
        traffic_point = PyKDL.Vector(traffic_x, traffic_y, traffic_z)
        
        Rotation = PyKDL.Rotation.Quaternion(*rot)
        Translation = PyKDL.Vector(*trans)
        traffic_p_car = Rotation*traffic_point + Translation
        rospy.loginfo('---------')
        rospy.loginfo(traffic_point)
        rospy.loginfo(traffic_p_car)
        
        f = 2350
        x_offset = 285
        y_offset = 455
        img_width  = 90
        img_height = 190

        x = int(-traffic_p_car[1]/traffic_p_car[0]*f + img_width/2 + x_offset)
        y = int(-traffic_p_car[2]/traffic_p_car[0]*f + img_height/2 + y_offset)
        x = 0 if (x < 0) else x
	y = 0 if (y < 0) else y

	xmax = x+img_width if(x+img_width <= cv_img.shape[1]-1) else cv_img.shape[1]-1
	ymax = y+img_height if(y+img_height <= cv_img.shape[0]-1) else cv_img.shape[0]-1
	cv_img = cv_img[y:ymax, x:xmax].copy()
        #cv2.rectangle(cv_img, (x,y), (x+img_width, y+img_height), (0,255,0),3)

	#base_dir = os.path.dirname(os.path.realpath(__file__))
        #img_name = base_dir + '/imgs/imgs' + str(self.picture_num) + '.jpg'
        #cv2.imwrite(img_name, cv_img)
        self.picture_num += 1
        return cv_img

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

	print 'test'
	rospy.loginfo('Get light state called: %s', self.has_image)

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cropped_camera_image = self.crop_image(light)

        return self.light_classifier.get_classification(cropped_camera_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        stop_line_positions = self.config["stop_line_positions"]
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0],line[1])

                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
            
	closest_light_x = closest_light.pose.pose.position.x
	car_x = self.pose.pose.position.x
	is_distance_valid = ((closest_light_x - car_x) < 100) and not self.is_site
	# when closest_light is far away don't keep calling crop and classifier
        if closest_light and is_distance_valid:
	    rospy.loginfo('Self Pose x: %s', self.pose.pose.position.x)
	    rospy.loginfo('Self Pose y: %s', self.pose.pose.position.y)
	    rospy.loginfo('Light Pose x: %s', closest_light.pose.pose.position.x)
	    rospy.loginfo('Light Pose y: %s', closest_light.pose.pose.position.y)
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

	return -1, TrafficLight.UNKNOWN
if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
