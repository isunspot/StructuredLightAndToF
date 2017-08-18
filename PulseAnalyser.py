from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import blackman
from scipy.fftpack import fft, ifft
from sklearn import decomposition


def smooth(y, box_pts):
    y_prev = y[0:box_pts+10]
    y_end = y[len(y)-box_pts-10]
    #y_short = y[box_pts + 10: len(y)-box_pts-10]

    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')

    y = y_smooth
    y[0:box_pts+10] = y_prev
    y[len(y)-box_pts-10:len(y)] = y_end

    return y

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
        self.__C1 = []
        self.__R = []
        self.__G = []
        self.__B = []

    # TODO get all channels
    def get_channels(self, avi_rgb_file, avi_grey_file):

        self.__avi_rgb = avi_rgb_file
        self.__avi_grey = avi_grey_file

        # Read first frames
        self.__cap_rgb = cv2.VideoCapture(avi_rgb_file)
        self.__cap_grey = cv2.VideoCapture(avi_grey_file)
        ret_rgb, frame_rgb1 = self.__cap_rgb.read()
        ret_grey, frame_grey1 = self.__cap_grey.read()

        # Mirror grey frame
        mirror_grey1 = cv2.flip(frame_grey1, 1)

        # TODO Non-contant arguments
        self.calculatePerspectiveTransform("out1.png", "out2.png")
        frame_rgb1 = self.transform(frame_rgb1, mirror_grey1)

        # Here we select ROI
        self.select_roi(frame_rgb1)
        cropped_grey1 = self.crop(frame_rgb1)
        i = 0

        while(1):
            ret_rgb, frame_rgb2 = self.__cap_rgb.read()
            ret_grey, frame_grey2 = self.__cap_grey.read()

            if(ret_rgb & ret_grey == True):

                mirror_grey2 = cv2.flip(frame_grey2, 1)
                frame_rgb2 = self.transform(frame_rgb2, mirror_grey2)

                # Crop current ROI
                cropped_rgb2 = self.crop(frame_rgb2)
                cropped_grey2 = self.crop(mirror_grey2)

                self.__C1.append(np.mean(cropped_grey2[: ,:, 1]))
                self.__R.append(np.mean(cropped_rgb2[:, :, 0]))
                self.__G.append(np.mean(cropped_rgb2[:, :, 1]))
                self.__B.append(np.mean(cropped_rgb2[:, :, 2]))

                i += 1
            else:
                break

        # Smoothing
        r = 3
        self.__C1 = smooth(self.__C1, r)
        self.__R = smooth(self.__R, r)
        self.__G = smooth(self.__G, r)
        self.__B = smooth(self.__B, r)

        plt.figure(1)
        plt.plot(self.__C1, label="pasmo szarości", color="black")
        plt.plot(self.__R, label="pasmo czerwieni", color="red")
        plt.plot(self.__G, label="pasmo zielone", color="green")
        plt.plot(self.__B, label="pasmo niebieskie", color="blue")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.title("Srednie wartosci z kanalow barwnych")
        plt.show(block=True)

        # dataset
        matrix = np.zeros([len(self.__C1), 4])
        for i in range(len(matrix)):
            matrix[i] = [self.__C1[i], self.__R[i], self.__G[i], self.__B[i]]
        data = np.mat(matrix)

        # PCA
        pca_components = 4
        pca = decomposition.PCA(n_components=pca_components).fit(data)
        X_out_pca = pca.transform(data)

        plt.figure(2)
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(X_out_pca[:,i])
            plt.title("PCA kanał " + str(i+1))
        plt.show(block=True)


        # ICA
        ica = decomposition.FastICA(n_components=4)
        ICA_out = ica.fit(X_out_pca).transform(X_out_pca)  # Estimate the sources

        plt.figure(3)
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.plot(ICA_out[:,i])
            plt.title("ICA kanał " + str(i+1))
        plt.show(block=True)

        #FFT of PCA
        for i in range(4):
            xf, yf = self.count_fft(X_out_pca[:, i])
            N = len(X_out_pca[:,i])

            for index, freq in enumerate(xf):
                if (freq < 0.04 or freq > 4):
                    yf[index] = 0
            plt.figure(4)
            plt.subplot(2,2,i+1)
            plt.plot(xf, np.abs(yf[0:N // 2]))
            plt.title("FFT pf PCA")

            #IFFT
            new_yf = ifft(yf)

            plt.figure(5)
            plt.subplot(2,2,i+1)
            plt.plot(new_yf)
            plt.title("IFFT pf PCA")

        plt.show(block=True)

        # FFT of ICA
        for i in range(4):
            xf, yf = self.count_fft(ICA_out[:, i])
            N = len(ICA_out[:, i])

            for index, freq in enumerate(xf):
                if (freq < 0.04 or freq > 4):
                    yf[index] = 0
            plt.figure(6)
            plt.subplot(2, 2, i + 1)
            plt.plot(xf, np.abs(yf[0:N // 2]))
            plt.title("FFT pf ICA")

            # IFFT
            new_yf = ifft(yf)

            plt.figure(7)
            plt.subplot(2, 2, i + 1)
            plt.plot(new_yf)
            plt.title("IFFT pf ICA")

        plt.show(block=True)

    def count_fft(self, y, time_spacing=0.04):

        N = len(y)
        yf = fft(y)

        # Windowing
        w = blackman(N)
        ywf = fft(y * w)

        T = time_spacing
        xf = np.linspace(0.0, 1.0 / (2 * T), len(y) // 2)

        return [xf, yf]


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
        #img_out2 = img2

        img_out1 = cv2.warpPerspective(img_in1, self.__h, (img_in1.shape[1], img_in1.shape[0]))
        return img_out1
        # cv2.imshow("1", img_out1)
        # cv2.imshow("2", img_out2)
        # cv2.waitKey()


#############

wavelength = 500
grey_filepath = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\PULS\\pomiary_PULS\\K_1_seria_G\\" +  str(wavelength) + "_G.avi")
rgb_filepath = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\PULS\\pomiary_PULS\\K_1_seria_RGB\\" + str(wavelength)  + "_RGB.avi")

pulseAnalyser = PulseAnalyser()
pulseAnalyser.get_channels(avi_rgb_file=rgb_filepath, avi_grey_file=grey_filepath)



