from matplotlib import pyplot as plt
import cv2
import os
import numpy as np


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class PulseAnalyser(object):

    def __init__(self):
        self.__avi_rgb = None

        # ROI in RGB frame
        self.__y1 = None
        self.__y2 = None
        self.__x1 = None
        self.__x2 = None

        # For perspective transform
        self.__h = None

        # Channels

    # TODO get all channels
    def get_channels(self, avi_rgb_file, avi_grey_file):

        self.__avi_rgb = avi_rgb_file
        self.__avi_grey = avi_grey_file

        # Read first frames
        self.__cap_rgb = cv2.VideoCapture(avi_rgb_file)
        self.__cap_grey = cv2.VideoCapture(avi_grey_file)
        ret_rgb, frame_rgb1 = self.__cap_rgb.read()
        ret_grey, frame_grey1 = self.__cap_grey.read()

        # Here we select ROI
        # self.select_roi(frame_rgb1)
        # cropped_rgb1 = self.crop(frame_rgb1)

        print(frame_grey1)
        print(frame_rgb1)
        print("----------")

        # Mirror grey frame
        mirror_grey1 = cv2.flip(frame_grey1, 1)

        # Here we are calculating tranfrom
        cv2.imshow("1", frame_rgb1)
        cv2.imshow("2", mirror_grey1)
        cv2.waitKey()

        self.calculatePerspectiveTransform("out1.png", "out2.png")


        ret_rgb, frame_rgb2 = self.__cap_rgb.read()
        ret_grey, frame_grey2 = self.__cap_grey.read()
        mirror_grey2 = cv2.flip(frame_grey1, 1)
        self.transform(frame_rgb2, mirror_grey2)


    def select_roi(self, frame):
        r = cv2.selectROI(frame, fromCenter=0)
        print("Zaznacz obszar do przycięcia!")
        print("-----------------")

        self.__y1 = int(r[1])
        self.__y2 = int(r[3] + r[1])
        self.__x1 = int(r[0])
        self.__x2 = int(r[0] + r[2])

    def calculatePerspectiveTransform(self, rgb_in_png, grey_in_png):
        img1 = cv2.imread(rgb_in_png)
        img2 = cv2.imread(grey_in_png)

        copy_of_img2 = img2

        orb = cv2.ORB_create(nfeatures=1000)

        # find the keypoints and descriptors with SIFT
        kp1 = orb.detect(img1, None)
        kp2 = orb.detect(img2, None)

        kp1, des1 = orb.compute(img1, kp1)
        kp2, des2 = orb.compute(img2, kp2)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)  # [m]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.__h = h

        # Presentation
        img_in1 = img1
        img_out1 = cv2.warpPerspective(img_in1, h, (img_in1.shape[1], img_in1.shape[0]))
        img_out2 = copy_of_img2

        cv2.imshow("1", img_out1)
        cv2.imshow("2", img_out2)
        cv2.waitKey()

    def crop(self, frame):
        return frame[self.__y1:self.__y2, self.__x1:self.__x2]

    def transform(self, img1, img2):

        img_in1 = img1
        img_out2 = img2

        img_out1 = cv2.warpPerspective(img_in1, self.__h, (img_in1.shape[1], img_in1.shape[0]))

        cv2.imshow("1", img_out1)
        cv2.imshow("2", img_out2)
        cv2.waitKey()


#############

wavelength = 500
grey_filepath = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\PULS\\pomiary_PULS\\K_1_seria_G\\" +  str(wavelength) + "_G.avi")
rgb_filepath = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\PULS\\pomiary_PULS\\K_1_seria_RGB\\" + str(wavelength)  + "_RGB.avi")

pulseAnalyser = PulseAnalyser()
pulseAnalyser.get_channels(avi_rgb_file=rgb_filepath, avi_grey_file=grey_filepath)



