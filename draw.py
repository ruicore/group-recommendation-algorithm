# @Author:             何睿
# @Create Date:        2019-05-24 12:18:33
# @Last Modified by:   何睿
# @Last Modified time: 2023-03-24 14:26:37

import matplotlib.pyplot as plt
import numpy

x_row = numpy.array([5, 10, 15, 20, 25, 30])

T1 = numpy.array([0.731, 0.728, 0.726, 0.712, 0.677, 0.648])
T2 = numpy.array([0.733, 0.730, 0.729, 0.713, 0.679, 0.649])
T3 = numpy.array([0.735, 0.731, 0.733, 0.711, 0.682, 0.651])

x = numpy.arange(5, 30)

l1 = plt.plot(x_row, T1, "ro-", label="0.4")
l2 = plt.plot(x_row, T2, "g+-", label="0.6")
l3 = plt.plot(x_row, T3, "b^-", label="0.8")

plt.plot(x_row, T1, "ro-", x_row, T2, "g+-", x_row, T3, "b^-")

plt.title("Threshold")
plt.xlabel("Group Size")
plt.ylabel("F")
plt.legend()
plt.show()
