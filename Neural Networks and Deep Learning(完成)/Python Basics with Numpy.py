test = "Hello World"
print("test: " +test)

import numpy as np

# def basic_sigmoid(x):
#
#     s = 1/(1+np.exp(-x))
#
#     return s
#
# print(basic_sigmoid(3))
#
# x = np.array([1,2,3])
# print(np.exp(x))
# print(x + 3)
# print(basic_sigmoid(x))
#
# def sigmod_derivative(x):
#
#     s = 1/(1+np.exp(-x))
#     ds = s*(1-s)
#
#     return ds
#
# x = np.array([1,2,3])
# print("sigmod_derivative(x)= " +str(sigmod_derivative(x)))
#
# def image2vector(image):
#
#     v = image.reshape((image.shape[1]*image.shape[2], 3))
#
#     return v
#
# image = np.array([[[ 0.67826139,  0.29380381],
#         [ 0.90714982,  0.52835647],
#         [ 0.4215251 ,  0.45017551]],
#
#        [[ 0.92814219,  0.96677647],
#         [ 0.85304703,  0.52351845],
#         [ 0.19981397,  0.27417313]],
#
#        [[ 0.60659855,  0.00533165],
#         [ 0.10820313,  0.49978937],
#         [ 0.34144279,  0.94630077]]])
# print("image2vector(image)= "+str(image2vector(image)))

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])

print(np.exp(x))
print(np.sum(x, axis=1, keepdims = True))