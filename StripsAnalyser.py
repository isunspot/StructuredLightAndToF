import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import decomposition
import h5py
import math
from scipy.fftpack import fft, ifft
import random_points as rp


class StripsAnalyser(object):

    def __init__(self):
        self.__cap = None
        self.__all_unwrapped = []

    def save(self, filename_save):
        with h5py.File(filename_save, 'w') as hf:
            hf.create_dataset("all_unwrapped", data=self.__all_unwrapped)

    def show(self, filename_save):
        with h5py.File(filename_save, 'r') as hf:
            self.__all_unwrapped = hf['all_unwrapped'][:]

        plt.subplot(1,2,1)
        plt.plot(self.__all_unwrapped)
        print(self.__all_unwrapped)
        plt.title("Unwrapped sum of phases")
        plt.xlabel("[n]")

        plt.subplot(1,2,2)
        N = len(self.__all_unwrapped)
        T = 0.03
        freq = np.linspace(1, 1/(2*T), N//2 )
        plt.plot(freq, np.abs(fft(self.__all_unwrapped[0:N//2])))
        plt.title("FFT of unwrapped sum of phases")
        plt.xlabel("f[Hz]")
        plt.show(block=True)

    def show_and_calc_and_save(self, filepath, file_to_save):

        self.__cap = cv2.VideoCapture(filepath)

        # # Take first frame and find corners in it
        ret, frame = self.__cap.read()

        if (ret==True):

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            N = 600 # liczba ramek
            h = len(frame) # wysokość ramki
            w = len(frame[1]) # szerokość ramki

            all_frames = np.empty([h, w, N])

            index = 0
            while(self.__cap.isOpened()):

                ret, frame = self.__cap.read()

                if (frame == None):
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                all_frames[:,:,index] = frame
                index += 1

                k = cv2.waitKey(30) & 0xff

                if k == ord('q'):
                    break

                if k == 27:
                    break

                if index >= N:
                    print(frame)
                    print(all_frames[:,:,index-1])
                    break

            lastIndex = index - 1
            cv2.destroyAllWindows()
            self.__cap.release()

            random_points = rp.random_points(number_of_points=10, max1=h, mu1=0.5*h, sigma1=h*0.1, max2=w, mu2=0.5*w, sigma2=w*0.1)

            for point in random_points:

                    signal_from_single_pxl = all_frames[500, 600, :lastIndex]

                    # FFT
                    N = len(signal_from_single_pxl)
                    T = 0.03

                    yf = fft(signal_from_single_pxl)
                    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

                    # Windowing
                    min_window_freq = 0.04
                    max_window_freq = 1.0
                    yf_window = np.copy(yf)

                    for index, element in enumerate(xf):
                        if element < min_window_freq or element > max_window_freq:
                            yf_window[index] = 0

                    # IFFT
                    new_ifft = ifft(yf_window)

                    # PHASE CALCULATING
                    arg = np.copy(new_ifft)

                    for index in range(len(new_ifft)):
                        if new_ifft[index].real != 0:
                            arg[index] = new_ifft[index].imag / new_ifft[index].real
                        else:
                            # TODO NAN when dividing by 0
                            arg[index] = 0

                    arg = arg.real

                    phase = np.arctan(arg)
                    phase_clone = np.copy(phase)

                    # UNWRAPPING
                    unwrapped = np.unwrap(phase_clone)

                    # SUM OF ALL PHASES
                    if len(self.__all_unwrapped) == 0:
                        self.__all_unwrapped = unwrapped

                    else:
                        self.__all_unwrapped = np.add(unwrapped, self.__all_unwrapped)

                    # PLOTTING

                    # plt.ion()
                    #
                    # plt.subplot(2, 3, 1)
                    # plt.plot(np.linspace(0, len(signal_from_single_pxl), num=len(signal_from_single_pxl)), signal_from_single_pxl)
                    # plt.title("Original signal from pxl")
                    #
                    # plt.subplot(2, 3, 2)
                    # plt.plot(xf, np.abs(yf[0:len(signal_from_single_pxl) // 2]))
                    # plt.title("FFT of original signal")
                    #
                    # plt.subplot(2, 3, 3)
                    # plt.plot(xf, np.abs(yf_window[0:len(signal_from_single_pxl) // 2]))
                    # plt.title("Filtered FFT")
                    #
                    # plt.subplot(2, 3, 4)
                    # plt.plot(np.linspace(0, len(new_ifft), num=len(new_ifft)), np.abs(new_ifft))
                    # plt.title("IFFT of filtered FFT")
                    #
                    # plt.subplot(2, 3, 5)
                    # plt.plot(np.linspace(0, len(phase), num=len(phase)), phase)
                    # plt.title("Phase")
                    #
                    # plt.subplot(2, 3, 6)
                    # plt.plot(np.linspace(0, len(phase), num=len(phase)), unwrapped)
                    # plt.title("Unwrapped")
                    #
                    # #plt.show(block=True)
                    #
                    # plt.pause(0.05)
                    # plt.gcf().clear()
                    #
                    # print("SAD" + str(sum(np.abs(phase-unwrapped))))

            #PLOT ALL POINTS ON ONE FRAME

            # SAVING
            self.save(file_to_save)
            print("Zapisano do " + str(file_to_save))

        else:
            print("Coś się nie udało :(")

#filename = "ramka_16_pion_1_10_ramek.avi"
filename = "ramka_16_pion_1.avi"
#filepath_avi = os.path.abspath("F:\\K&A_pomiary\\2_kamery_basler_see\\paski\\" + filename)
filepath_avi = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\SPECKLE\\paski\\" + filename)

filename_save = "strips_16_one_point.h5"

strips = StripsAnalyser()
strips.show_and_calc_and_save(filepath_avi, filename_save)

strips.show(filename_save)