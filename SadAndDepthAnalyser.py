import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import decomposition


import h5py
from scipy import signal
from scipy.signal import blackman
from scipy.fftpack import fft, ifft
import math

# This class is responsible for calculating SAD (sum of absolute differences)
# and also a mean intensity (this intensity can be a function of depth if is so).

# It is dedicated for avi files with structured light visualisation movements.
# It can be used for avi depth files.

class SadAndDepthAnalyser(object):

    def __init__(self):
        self.__cap = None
        self.__allSad = []
        self.__allCcr = []
        self.__allDistances = []
        self.__y1 = 0
        self.__y2 = 0
        self.__x1 = 0
        self.__x2 = 0

    def save_sad(self, filename_save):
        with h5py.File(filename_save, 'w') as hf:
             hf.create_dataset("allSad", data=self.__allSad)

    def read_sad(self, filename_read):
        with h5py.File(filename_read, 'r') as hf:
            self.__allSad = hf['allSad'][:]

        self.plot_sad()

    def save_dist(self, filename_save):
        with h5py.File(filename_save, 'w') as hf:
            hf.create_dataset("allDistances", data=self.__allDistances)

    def read_dist(self, filename_read):
        with h5py.File(filename_read, 'r') as hf:
            self.__allSad = hf['allDistances'][:]

        self.plot_distances()

    def select_roi(self, frame):
        r = cv2.selectROI(frame, fromCenter=0)
        print("Zaznacz ROI!")
        print("-----------------")

        self.__y1 = int(r[1])
        self.__y2 = int(r[3] + r[1])
        self.__x1 = int(r[0])
        self.__x2 = int(r[0] + r[2])

    def correlation_coefficient(patch1, patch2):
        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            product /= stds
        return product


    def calc_sad(self, old_frame, new_frame):

        current_sad = cv2.absdiff(old_frame, new_frame)
        return np.sum(current_sad)

    def count_fft(self):

        y = self.__allSad
        N = len(self.__allSad)

        #Windowing
        w = blackman(N)
        ywf = fft(y * w)

        #yf = fft(y)
        T = 0.04
        xf = np.linspace(0.0, 1.0/(2.0*T), len(y)//2)

        print(xf)
        print(ywf)

        print(len(xf))
        print(len(ywf))

        return [xf, ywf]

    def calc_ccor(self, old_frame, new_frame):

        current_ccr = signal.correlate2d(old_frame, new_frame)
        return np.sum(current_ccr)

    def calc_distances(self, frame_cropped):

        return np.mean(frame_cropped)


    def calc(self, filepath):

            self.__cap = cv2.VideoCapture(filepath)

            # Take first frame and find corners in it
            ret, old_frame = self.__cap.read()

            cv2.imshow("Frame", old_frame)

            #self.select_roi(old_frame)
            #old_frame = old_frame[self.__y1:self.__y2, self.__x1:self.__x2]


            if (ret == True):


                old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

                #counter = 0
                while (self.__cap.isOpened()):

                    ret, frame = self.__cap.read()

                    #counter += 1
                    #if (counter >= 5):
                    #    break

                    if (ret == True):
                        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        cv2.imshow("Frame", new_frame)

                        #new_frame = new_frame[self.__y1:self.__y2, self.__x1:self.__x2]

                        # Calculatuing the SAD
                        current_sad_sum = self.calc_sad(old_frame, new_frame)
                        self.__allSad.append(current_sad_sum)

                        # Calculating the sum of crosscorelation
                        # current_ccr_sum = self.calc_ccor(old_frame, new_frame)
                        # self.__allCcr.append(current_ccr_sum)

                        # Calculating distanses
                        current_distance = self.calc_distances(new_frame)
                        self.__allDistances.append(current_distance)

                        k = cv2.waitKey(30) & 0xff
                        if k == 27:
                            break

                        # Now update the previous frame and previous points
                        old_frame = new_frame.copy()

                    else:
                        break


                cv2.destroyAllWindows()
                self.__cap.release()

            else:
                print("Coś się nie udało :(")

    def plot_sad(self):
            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(self.__allSad)
            plt.title("SAD")

            fft_all = self.count_fft()
            xf = fft_all[0]
            yf = fft_all[1]

            N = len(self.__allSad)
            plt.subplot(2,2,2)
            plt.semilogy(xf, np.abs(yf[0:N//2]))
            #2.0/N*np.abs(yf[0:N//2]))


            plt.show()

    def plot_distances(self):
            plt.plot(self.__allDistances)
            plt.title("Mean Intensity (or Distances)")
            plt.show()

    def plot_ccr(self):
            plt.plot(math.ln(self.__allCcr + 1))
            plt.title("Sum Of Cross-Correlation")
            plt.show()


filename = "ramka_100.avi"
filepath_avi = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\2_kamery_basler_see\\100cm\\BaslerUSB\\" + filename)


sad_reader = SadAndDepthAnalyser()
sad_reader.calc(filepath_avi)
sad_reader.plot_sad()



#filepath_avi = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\pylon_spekle\\" + filename)
#filename_save = "pylon_spekle_sad.h5"
#sad_reader.save_sad(filename_save)
#sad_reader.plot_distances()
#sad_reader.plot_ccr()
#sad_reader.read(filename_save)