"""
Created on Wed Mar 16 11:37:50 2022

@author: okritvik
"""

import numpy as np

# Camera Coordinates
cc = [(758,686),(759,966),(1190,172),(329,1041),(1204,850),(340,159)]

# World Coordinates
wc = [(0,7,0),(0,11,0),(7,1,0),(0,11,7),(7,9,0),(0,1,7)]

#A Matrix
A = []
for i in range(0, len(cc)):
    u,v = cc[i]
    x,y,z = wc[i]
    A.append([0, 0, 0, 0, -x, -y, -z, -1, v*x, v*y, v*z, v])
    A.append([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]) 
    #Computing the A Matrix as given in the slides to get projection matrix computation

print("A Matrix:")
print(A)

A = np.asarray(A)

print(A.shape)

# SVD to compute the Projection matrix
U,S,Vt = np.linalg.svd(A)

P = Vt[-1,:] / Vt[-1,-1] #Normalizing

req_P = np.reshape(P,(3,4))
print("Projection Matrix: ")
print(req_P)

#Getting the left 3x3 matrix of the Projection matrix to do QR factorization
M = []

M.append(req_P[:,0])
M.append(req_P[:,1])
M.append(req_P[:,2])

M = np.asarray(M).T
# print(M)

R,K = np.linalg.qr(M)

K = K / K[-1,-1]
print("The Camera Intrinsic Parameters Matrix is: ")
print(K)

print("The Rotation Matrix: ")
print(R)

print("The Translation Vector is ")
print(np.resize(req_P[:,3].T,(3,1)))