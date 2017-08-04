# Reading mat files
import scipy.io
import numpy as np

from matplotlib import pyplot as plt
import cv2

mat = scipy.io.loadmat('images/24_07/K_1_25.mat')
mat_array = mat['allDistances']

# Wartosci
frame_index = 0

# Wyciągam ramkę i rysuję

frame = mat_array[frame_index]

plt.imsave('outfile.png', frame)
z_img = cv2.imread('outfile.png')

#selecting ROI && cropping the image
r = cv2.selectROI(z_img, fromCenter=0)
print("Zaznacz ROI!")
print("-----------------")

y1 = int(r[1])
y2 = int(r[3] + r[1])
x1 = int(r[0])
x2 = int(r[0] + r[2])

img_crop = z_img[y1:y2,x1:x2]

# plt.imshow(img_crop, interpolation='none')
# plt.show()

print("Type of img crop: ")
print(z_img)
print(frame)

print("r: ")
print(r)

array_distance = np.empty([len(mat_array)])

frame_cropped = frame[y1:y2,x1:x2]

plt.imshow(frame_cropped, interpolation='none')
plt.colorbar()
plt.show()

for index_frame, frame in enumerate(mat_array):
    frame[frame >= 65535] = 300
    frame_cropped = frame[y1:y2, x1:x2]

    #Counting mean distance
    print("Średnia odległość klatki piersiowej dla klatki " + str(index_frame))
    array_distance[index_frame] = np.mean(frame_cropped)

t_stamp = 0.25 # [s]
print(array_distance)
plt.plot(np.arange(0, len(mat_array)*t_stamp, t_stamp), array_distance, 'm-')
plt.show()

