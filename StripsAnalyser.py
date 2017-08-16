import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import decomposition
import h5py
import math
from scipy.fftpack import fft, ifft


class StripsAnalyser(object):

    def __init__(self):
        self.__cap = None
        self.__all = []

    def save(self,filename_save):
        with h5py.File(filename_save, 'w') as hf:
            hf.create_dataset("all", data=self.__all)

    def show(self, filename_save):
        with h5py.File(filename_save, 'r') as hf:
            self.__all = hf['all'][:]

    def show_and_calc_and_save(self, filepath, file_to_save):

        self.__cap = cv2.VideoCapture(filepath)

        # # Take first frame and find corners in it
        ret, frame = self.__cap.read()

        if (ret==True):

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            N = 600
            h = len(frame)
            w = len(frame[1])

            all_frames = np.empty([h, w, N])

            index = 0
            while(self.__cap.isOpened()):

                ret, frame = self.__cap.read()

                if (frame == None):
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #cv2.imshow('frame', frame)

                all_frames[:,:,index] = frame
                index += 1

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                if index >= N:
                    print(frame)
                    print(all_frames[:,:,index-1])
                    break

            lastIndex = index - 1

            cv2.destroyAllWindows()
            self.__cap.release()
            self.save(file_to_save)
            self.show(file_to_save)

            # Plot
            signal_from_single_pxl = all_frames[500,600,:lastIndex]
            # plt.plot(signal_from_single_pxl)
            # plt.show(block=True)

            # Count fft
            N = len(signal_from_single_pxl)
            T = 0.03

            yf = fft(signal_from_single_pxl)
            xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

            min_window_freq = 0.04
            max_window_freq = 0.5

            # Windowing
            yf_window = np.copy(yf)

            for index, element in enumerate(xf):
                if element < min_window_freq or element > max_window_freq:
                    yf_window[index] = 0

            # IFFT
            new_ifft = ifft(yf_window)

            # PHASE CALCULATING
            arg = np.divide(new_ifft.imag, new_ifft.real)
            phase = np.arctan(arg)

            # PLOTTING
            plt.subplot(2,3,1)
            plt.plot(signal_from_single_pxl)

            plt.subplot(2, 3, 2)
            plt.plot(xf, np.abs(yf[0:len(signal_from_single_pxl) // 2]))

            plt.subplot(2, 3, 3)
            plt.plot(xf, np.abs(yf_window[0:len(signal_from_single_pxl) // 2]))

            plt.subplot(2, 3, 4)
            plt.plot(new_ifft)

            plt.subplot(2, 3, 5)
            plt.plot(phase)

            plt.show()

        else:
            print("Coś się nie udało :(")



filename = "ramka_16_pion_1.avi"
filepath_avi = os.path.abspath("F:\\K&A_pomiary\\2_kamery_basler_see\\paski\\" + filename)

filename_save = "stereo.h5"

strips = StripsAnalyser()
strips.show_and_calc_and_save(filepath_avi, filename_save)
#speckles_reader.show(filename_save)