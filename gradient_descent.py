import numpy as np

def gradient_descent(x, lrn_rte = 0.00001):
   count = 0
   slope = 2 * x

   while np.round(slope, 2) != 0:
       x = x - lrn_rte * slope
       slope = 2 * x
       count += 1
       print("Number of iterations to converge : ", count)
       print("Present position of x: ", x)
       print("Present value of slope: ", slope)
       print("--------------------------------")