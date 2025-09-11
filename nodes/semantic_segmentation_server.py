#!/usr/bin/env python3

import cv_bridge
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image

from semantic_segmentation_ros.detection import SemanticSegmentation
from semantic_segmentation_ros.rviz import Visualizer
from semantic_segmentation_ros.srv import GetSegmentedImage, GetSegmentedImageResponse


class SemanticSegmentationServer:
    def __init__(self) -> None:
        """
        Initialize the ROS node, load parameters, set up publishers, subscribers, and services.
        Also, initialize the segmentation model and visualizer with the loaded parameters.

        Inputs: None    
        Outputs: None
        """
        self.load_parameters()
        self.init_pubsub()
        self.init_services()
        self.latest_mask_pred1 = None
        self.latest_mask_pred2 = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_bridge = cv_bridge.CvBridge()
        self.segmentation_model1 = SemanticSegmentation(self.model_name, self.encoder_name, self.encoder_weights, self.in_channels, self.classes, self.model_path1, self.device)
        self.segmentation_model2 = SemanticSegmentation(self.model_name, self.encoder_name, self.encoder_weights, self.in_channels, self.classes, self.model_path2, self.device)
        self.vis = Visualizer(self.classes)

###########################################################################
        # try:
        #     self.segmentation_model = SemanticSegmentation(
        #         self.model_name,
        #         self.encoder_name,
        #         self.encoder_weights,
        #         self.in_channels,
        #         self.classes,
        #         self.model_path,
        #         self.device
        #     )
        # except Exception as e:
        #     rospy.logerr(f"Model load failed: {e}")
        #     self.segmentation_model = None
##########################################################################
        rospy.loginfo("Semantic Segmentation Server is ready")

    def load_parameters(self) -> None:
        """
        Load parameters from the ROS parameter server necessary for the node's operation, including topics, model details, and paths.

        Inputs: None
        Outputs: None
        """
        self.color_topic = rospy.get_param("~camera/color_topic")
        self.model_name = rospy.get_param("~model/model_name")
        self.encoder_name = rospy.get_param("~model/encoder_name")
        self.encoder_weights = rospy.get_param("~model/encoder_weights")
        self.in_channels = rospy.get_param("~model/in_channels")
        self.classes = rospy.get_param("~model/classes")
        self.model_path1 = rospy.get_param("~model/model_path1")
        self.model_path2 = rospy.get_param("~model/model_path2")

    def init_pubsub(self) -> None:
        """
        Initialize ROS publishers and subscribers. Sets up a publisher for segmentation masks and subscribes to the color image topic.

        Inputs: None
        Outputs: None
        """
        self.segmentation_mask_pub1 = rospy.Publisher("segmentation_mask1", Image, queue_size=1)
        self.segmentation_mask_pub2 = rospy.Publisher("segmentation_mask2", Image, queue_size=1)
        #self.combined_segmentation_mask_pub = rospy.Publisher("combined_segmentation_mask", Image, queue_size=1)
        rospy.Subscriber(self.color_topic, Image, self.rgb_image_callback)

    def init_services(self) -> None:
        """
        Initialize ROS services to allow other nodes to request the segmented mask.

        Inputs: None
        Outputs: None
        """
        rospy.Service("get_segmentation_image", GetSegmentedImage, self.get_segmentation_image)

    def rgb_image_callback(self, msg: Image) -> None:
        """
        Callback function for the color image subscriber.
        Converts ROS image messages to CV2 image arrays, processes them through the segmentation model, and publishes the resulting mask and segmented images.

        Inputs: msg (Image) - The received image message
        Outputs: None
        """
        try:
            mask_pred1 = self.segmentation_model1.predict(self.cv_bridge.imgmsg_to_cv2(msg, "rgb8").transpose(2, 0, 1).astype(np.float32))
            mask_pred2 = self.segmentation_model2.predict(self.cv_bridge.imgmsg_to_cv2(msg, "rgb8").transpose(2, 0, 1).astype(np.float32))
            # publish mask
            self.latest_mask_pred1 = self.cv_bridge.cv2_to_imgmsg(mask_pred1.astype(np.uint8), "mono8")
            self.segmentation_mask_pub1.publish(self.latest_mask_pred1)
            self.latest_mask_pred2 = self.cv_bridge.cv2_to_imgmsg(mask_pred2.astype(np.uint8), "mono8")
            self.segmentation_mask_pub2.publish(self.latest_mask_pred2)
            ###ここで２つのマスクを統合させて一つのマスクにする処理を入れる
            #H*W*２
            combined_mask = np.stack((mask_pred1, mask_pred2), axis=-1)
            ##u,v,0がmask1、u,v,1がmask2
            self.latest_combined_mask = self.cv_bridge.cv2_to_imgmsg(combined_mask.astype(np.uint8), "8UC2")###これで２チャンネルとなる
            #self.combined_segmentation_mask_pub.publish(self.latest_combined_mask)
            # publish image
            self.vis.publish_segmented_image1(mask_pred1)
            self.vis.publish_segmented_image2(mask_pred2)
            #vals = np.unique(mask_pred)
            #rospy.loginfo(f"Segmentation Done. Unique labels in mask: {vals}")
        except Exception as e:
            rospy.logerr(f"Failed to Process Image : {e}")

    def get_segmentation_image(self, req) -> GetSegmentedImageResponse:
        """
        Service handler to provide the latest segmented image. Returns an empty response if no segmented image is available.

        Inputs: req - The service request
        Outputs: GetSegmentedImageResponse - The response containing the segmented image
        """
        if (self.latest_mask_pred1 is None) or (self.latest_mask_pred2 is None):
            rospy.logwarn("No Segmented Image Available")
            return GetSegmentedImageResponse()
        return GetSegmentedImageResponse(self.latest_mask_pred1)
#,GetSegmentedImageResponse(self.latest_mask_pred2)
    

if __name__ == "__main__":
    rospy.init_node("semantic_segmentation_server")
    server = SemanticSegmentationServer()
    rospy.spin()
