import cv2
import numpy as np
import os
import h5py
from matplotlib import pyplot as plt

# This class is responsible for analysing optical flow density
# and plots changes of value and hue.

# I takes as an input an avi files.

class OpticalFlowDensityAnalyser(object):

    def __init__(self):
        self.__cap = None
        self.__number_of_frames = None
        self.__allHueValue = None

        self.__y1 = None
        self.__y2 = None
        self.__x1 = None
        self.__x2 = None

    def save(self, filename_save):
        with h5py.File(filename_save, 'w') as hf:
            hf.create_dataset("all_frames", data=self.__allHueValue)

    def read(self, filename_read):
        with h5py.File(filename_read, 'r') as hf:
            self.__allHueValue = hf['all_frames'][:]
        self.plot()

    def plot(self):

        mean_hue = self.__allHueValue[:, 0:1]
        mean_value = self.__allHueValue[:, 1:2]

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(mean_hue)
        plt.title("Zmiany średniej wartości H")

        plt.subplot(2, 1, 2)
        plt.plot(mean_value)
        plt.title("Zmiany średniej wartości V")

        plt.show()

    def select_roi(self, frame):
        r = cv2.selectROI(frame, fromCenter=0)
        print("Zaznacz obszar do przycięcia!")
        print("-----------------")

        self.__y1 = int(r[1])
        self.__y2 = int(r[3] + r[1])
        self.__x1 = int(r[0])
        self.__x2 = int(r[0] + r[2])

    def crop(self, frame):
        return frame[self.__y1:self.__y2, self.__x1:self.__x2]

    def analyse(self, number_of_frames, filepath_avi, crop_roi=True):

        # First frame
        self.__cap = cv2.VideoCapture(filepath_avi)
        self.__number_of_frames = number_of_frames
        self.__allHueValue = np.empty((self.__number_of_frames, 2))

        ret, frame1 = self.__cap.read()

        if crop_roi==True:
            self.select_roi(frame1)
            frame1 = self.crop(frame1)

        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        cv2.imshow('frame1', frame1)
        cv2.waitKey()

        index_frame = 0

        while(True):

            ret, frame = self.__cap.read()

            if (ret == True):

                next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                if crop_roi == True:
                    next = self.crop(next)

                # Calc Optical Flow
                winsize = 30
                poly_n = 7
                poly_sigma = 1.5
                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, winsize, 3, poly_n, poly_sigma, 0)

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

                bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                cv2.imshow('frame2',bgr)

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                elif k == ord('s'):
                    cv2.imwrite('opticalfb.png', next)
                    cv2.imwrite('opticalhsv.png', bgr)

                prvs = next

                if (index_frame < self.__number_of_frames):
                    self.__allHueValue[index_frame, 0] = np.mean(bgr[:,:,0])
                    self.__allHueValue[index_frame, 1] = np.mean(bgr[:,:,2])

                    index_frame = index_frame + 1

                else:
                    break



        self.__cap.release()
        cv2.destroyAllWindows()


flow_analyser = OpticalFlowDensityAnalyser()

filename_avi = "4_oddech_przykrywka.avi"
filepath_avi = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\pylon_spekle\\" + filename_avi)

#
#
# flow_analyser.analyse(number_of_frames=10, filepath_avi=filepath_avi, crop_roi=False)
# flow_analyser.save(filename_save="optical_flow_files/all_frames_1.h5")
# flow_analyser.plot()


flow_analyser.read(filename_read="optical_flow_files/all_frames_hue_value.h5")






