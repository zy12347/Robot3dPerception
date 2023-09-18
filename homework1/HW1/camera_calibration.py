import cv2
import os
import sys
import numpy as np

def calculate_projection(pts2d, pts3d):
    """
    Compute a 3x4 projection matrix M using a set of 2D-3D point correspondences.

    Given a set of N 2D image points (pts2d) and their corresponding 3D world coordinates
    (pts3d), this function calculates the projection matrix M using the Direct Linear
    Transform (DLT) method. The projection matrix M relates the 3D world coordinates to
    their 2D image projections in homogeneous coordinates.

    Parameters:
    pts2d (numpy.ndarray): An Nx2 array containing the 2D image points.
    pts3d (numpy.ndarray): An Nx3 array containing the corresponding 3D world coordinates.

    Returns:
    M (numpy.ndarray): A 3x4 projection matrix M that relates 3D world coordinates to 2D
                   image points in homogeneous coordinates.
    """
    N = len(pts3d)
    homo3d = np.concatenate((pts3d,np.ones((N,1))),axis=1)
    A = np.zeros((2*N,12))
    A[0::2,8:] = pts2d[:,1].reshape((N,1))*homo3d
    A[0::2,4:8] = -homo3d
    A[1::2,0:4] = -homo3d
    A[1::2,8:] = pts2d[:,0].reshape((N,1))*homo3d
    U,S,VT = np.linalg.svd(A)
    M = VT[-1].reshape(3,4)
    M = M/M[2,3]
    print(M)
    ####################################
    ##########YOUR CODE HERE############
    ####################################

    ####################################
    return M


def calculate_reprojection_error(pts2d,pts3d):
    """
    Calculate the reprojection error for a set of 2D-3D point correspondences.

    Given a set of N 2D image points (pts2d) and their corresponding 3D world coordinates
    (pts3d), this function calculates the reprojection error. The reprojection error is a
    measure of how accurately the 3D points project onto the 2D image plane when using a
    projection matrix.

    Parameters:
    pts2d (numpy.ndarray): An Nx2 array containing the 2D image points.
    pts3d (numpy.ndarray): An Nx3 array containing the corresponding 3D world coordinates.

    Returns:
    float: The reprojection error, which quantifies the accuracy of the 3D points'
           projection onto the 2D image plane.
    """
    M = calculate_projection(pts2d, pts3d)
    N = len(pts2d)
    homo3d = np.concatenate((pts3d,np.ones((N,1))),axis=1)
    projected = M.dot(homo3d.T)
    projected = projected/projected[-1]
    error = 1/N * np.linalg.norm(projected[0:2]-pts2d.T)
    print(error)
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    ####################################
    return error


if __name__ == '__main__':
    data = np.load("data/camera_calib_data.npz")
    pts2d = data['pts2d']
    pts3d = data['pts3d']

    P = calculate_projection(pts2d,pts3d)
    reprojection_error = calculate_reprojection_error(pts2d, pts3d)

    print(f"Projection matrix: {P}")    
    print()
    print(f"Reprojection Error: {reprojection_error}")