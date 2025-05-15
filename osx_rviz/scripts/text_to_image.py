#! /usr/bin/env python

"""
Text to Image Converter with Matrix-style Effects

This ROS node converts text messages into images with Matrix-style visual effects.
It subscribes to text messages and publishes them as images with green Matrix-like
styling, including glow effects, falling characters, and digital noise.

Subscribers:
    /osx_text_to_image (std_msgs/String): Receives text to convert to image

Publishers:
    /osx_status_text_image (sensor_msgs/Image): Publishes the converted image

The node creates a Matrix-style visualization with:
- Green text with glow effects
- Random falling characters in the background
- Digital noise effects
- Automatic font scaling to fit the image

Author: Cristian C. Beltran-Hernandez, Felix von Drigalski
"""

# Software License Agreement (BSD License)
#
# Copyright (c) 2021, OMRON SINIC X
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Team osx nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Cristian C. Beltran-Hernandez, Felix von Drigalski

import rospy
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TextImageWriter():
    """
    A class that handles the conversion of text messages to Matrix-style images.

    This class subscribes to text messages and converts them into images with
    Matrix-style visual effects. It manages the ROS communication and image
    processing to create the final visualization.
    """

    def __init__(self):
        """
        Initialize the TextImageWriter.

        Sets up the ROS subscribers and publishers, and initializes the image
        processing components.
        """
        self.img = np.zeros((100, 1000, 3), np.uint8)
        self.sub_text = rospy.Subscriber("/osx_text_to_image", String, self.write_text_and_pub)
        self.pub_img = rospy.Publisher("/osx_status_text_image", Image, queue_size=1)
        self.bridge = CvBridge()
        rospy.sleep(1.0)

    def get_optimal_font_scale(self, text, image_width, image_height):
        """
        Calculate the optimal font scale for the given text and image dimensions.

        Args:
            text (str): The text to be displayed
            image_width (int): Width of the target image
            image_height (int): Height of the target image

        Returns:
            float: The optimal font scale that fits the text within the image
        """
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                       fontScale=scale/10.0, thickness=1)
            text_width = textSize[0][0]
            text_height = textSize[0][1]
            if (text_width <= image_width - 20) and (text_height <= image_height - 40):
                return scale/10.0
        return 1

    def add_glow_effect(self, img, text, position, font, font_scale, thickness):
        """
        Add a glowing effect to the text.

        Creates a glowing effect by drawing the text multiple times with
        increasing blur and transparency.

        Args:
            img (numpy.ndarray): The input image
            text (str): The text to add glow to
            position (tuple): (x, y) coordinates for text position
            font: OpenCV font type
            font_scale (float): Font scale factor
            thickness (int): Thickness of the text

        Returns:
            numpy.ndarray: Image with glowing text effect
        """
        # Create a slightly larger image for the glow effect
        glow_img = np.zeros_like(img)

        # Draw the text multiple times with increasing blur to create glow
        for i in range(5, 0, -1):
            blur_size = i * 2 + 1
            temp_img = np.zeros_like(img)
            cv2.putText(temp_img, text, position, font, font_scale,
                        (0, 255, 0), thickness, cv2.LINE_AA)
            temp_img = cv2.GaussianBlur(temp_img, (blur_size, blur_size), 0)
            glow_img = cv2.addWeighted(glow_img, 1.0, temp_img, 0.3, 0)

        # Add the main text
        cv2.putText(glow_img, text, position, font, font_scale,
                    (0, 255, 0), thickness, cv2.LINE_AA)

        return glow_img

    def write_text_and_pub(self, text_message):
        """
        Process incoming text message and publish it as an image.

        This is the callback function for the text subscriber. It creates
        a Matrix-style visualization of the text and publishes it as an image.

        Args:
            text_message (std_msgs/String): The incoming text message
        """
        text = text_message.data
        rospy.loginfo("Received text to convert: " + text)

        # Create a black background
        self.img = np.zeros(self.img.shape, np.uint8)

        # Add some random matrix-style falling characters in the background
        for _ in range(20):
            x = np.random.randint(0, self.img.shape[1])
            y = np.random.randint(0, self.img.shape[0])
            char = chr(np.random.randint(33, 127))
            cv2.putText(self.img, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 100, 0), 1, cv2.LINE_AA)

        # Text parameters
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 2
        font_scale = self.get_optimal_font_scale(text, self.img.shape[1], self.img.shape[0])

        # Get text size for centering
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        # Calculate center position
        x = (self.img.shape[1] - text_width) // 2
        y = (self.img.shape[0] + text_height) // 2

        position = (x, y)

        # Add the main text with glow effect
        self.img = self.add_glow_effect(self.img, text, position, font, font_scale, thickness)

        # Add some noise to make it look more like the Matrix
        # noise = np.random.normal(0, 10, self.img.shape).astype(np.uint8)
        # self.img = cv2.add(self.img, noise)

        rospy.sleep(0.4)  # Required for the image to be filled

        # Publish the image
        self.imgmsg = self.bridge.cv2_to_imgmsg(self.img, encoding='passthrough')
        self.pub_img.publish(self.imgmsg)
        rospy.loginfo("Published image.")


if __name__ == "__main__":
    """
    Main entry point for the text to image converter node.

    Initializes the ROS node and starts the TextImageWriter.
    """
    rospy.init_node("osx_text_to_image_converter")
    c = TextImageWriter()
    rospy.loginfo("osx text to image writer started up")
    rospy.spin()
