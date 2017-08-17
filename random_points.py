import numpy as np
from numpy import random
from numpy import rint


def random_points(number_of_points, max1, max2, mu1=550, sigma1=60, mu2=550, sigma2=60):

    all_points = np.empty([number_of_points, 2])
    index = 0

    while(index < number_of_points):
        x = random.normal(mu1, sigma1)

        y = random.normal(mu2, sigma2)

        if x > max1 or y > max2:
            continue
        else:
            all_points[index, :] = [x, y]
            index += 1

    return rint(all_points)

