import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as _helper




def compute_fundamental_matrix(pts1, pts2, scale):
    """
    Compute the Fundamental matrix from corresponding 2D points in two images.

    Given two sets of corresponding 2D image points from Image 1 (pts1) and Image 2 (pts2),
    as well as a scaling factor (scale) representing the maximum dimension of the images, 
    this function calculates the Fundamental matrix.

    Parameters:
    pts1 (numpy.ndarray): An Nx2 array containing 2D points from Image 1.
    pts2 (numpy.ndarray): An Nx2 array containing 2D points from Image 2, corresponding to pts1.
    scale (float): The maximum dimension of the images, used for scaling the Fundamental matrix.

    Returns:
    F (numpy.ndarray): A 3x3 Fundamental matrix 
    """
    N = pts1.shape[0]
    pts1_mean = np.mean(pts1,0)
    pts2_mean = np.mean(pts2,0)
    pts1 = np.concatenate((pts1,np.ones((N,1))),axis=1)
    pts2 = np.concatenate((pts2,np.ones((N,1))),axis=1)
    T1 = np.identity(3)
    T1[0,2] = -pts1_mean[0]
    T1[1,2] = -pts1_mean[1]
    T1 = T1/scale
    
    T2 = np.identity(3)
    T2[0,2] = -pts2_mean[0]
    T2[1,2] = -pts2_mean[1]
    T2 = T2/scale
    pts1_normalized = T1.dot(pts1.T).T
    pts2_normalized = T2.dot(pts2.T).T
    A = np.ones((N,9))
    A[:,0] = pts1[:,0]*pts2[:,0]
    A[:,1] = pts1[:,0]*pts2[:,1]
    A[:,2] = pts1[:,0]
    A[:,3] = pts1[:,1]*pts2[:,0]
    A[:,4] = pts1[:,1]*pts2[:,1]
    A[:,5] = pts1[:,1]
    A[:,6] = pts2[:,0]
    A[:,7] = pts2[:,1]
    U,S,VT = np.linalg.svd(A)
    F_ = VT[-1,:].reshape((3,3))
    U_F,S_F,VT_F = np.linalg.svd(F_)
    F = U_F[:,0:2].dot(np.diag(S_F[0:2])).dot(VT_F[0:2,:])
    F_denorm = (T1.T).dot(F).dot(T2)
    ####################################
    ##########YOUR CODE HERE############
    ####################################

    ####################################
    return F_denorm

def compute_epipolar_correspondences(img1, img2, pts1, F):
    """
    Compute epipolar correspondences in Image 2 for a set of points in Image 1 using the Fundamental matrix.

    Given two images (img1 and img2), a set of 2D points (pts1) in Image 1, and the Fundamental matrix (F)
    that relates the two images, this function calculates the corresponding 2D points (pts2) in Image 2.
    The computed pts2 are the epipolar correspondences for the input pts1.

    Parameters:
    img1 (numpy.ndarray): The first image containing the points in pts1.
    img2 (numpy.ndarray): The second image for which epipolar correspondences will be computed.
    pts1 (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates img1 and img2.

    Returns:
    pts2_ep (numpy.ndarray): An Nx2 array of corresponding 2D points (pts2) in Image 2, serving as epipolar correspondences
                   to the points in Image 1 (pts1).
    """
    pts2_ep = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
   
    ####################################
    return pts2_ep

def compute_essential_matrix(K1, K2, F):
    """
    Compute the Essential matrix from the intrinsic matrices and the Fundamental matrix.

    Given the intrinsic matrices of two cameras (K1 and K2) and the 3x3 Fundamental matrix (F) that relates
    the two camera views, this function calculates the Essential matrix (E).

    Parameters:
    K1 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 1.
    K2 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 2.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates Camera 1 and Camera 2.

    Returns:
    E (numpy.ndarray): The 3x3 Essential matrix (E) that encodes the essential geometric relationship between
                   the two cameras.

    """
    E = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    ####################################
    return E 

def triangulate_points(E, pts1_ep, pts2_ep):
    """
    Triangulate 3D points from the Essential matrix and corresponding 2D points in two images.

    Given the Essential matrix (E) that encodes the essential geometric relationship between two cameras,
    a set of 2D points (pts1_ep) in Image 1, and their corresponding epipolar correspondences in Image 2
    (pts2_ep), this function calculates the 3D coordinates of the corresponding 3D points using triangulation.

    Extrinsic matrix for camera1 is assumed to be Identity. 
    Extrinsic matrix for camera2 can be found by cv2.decomposeEssentialMat(). Note that it returns 2 Rotation and 
    one Translation matrix that can form 4 extrinsic matrices. Choose the one with the most number of points in front of 
    the camera.

    Parameters:
    E (numpy.ndarray): The 3x3 Essential matrix that relates two camera views.
    pts1_ep (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    pts2_ep (numpy.ndarray): An Nx2 array of 2D points in Image 2, corresponding to pts1_ep.

    Returns:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point.
    point_cloud_cv (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point calculated using cv2.triangulate
    """
    point_cloud = None
    point_cloud_cv = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    ####################################
    return point_cloud, point_cloud_cv


def visualize(point_cloud):
    """
    Function to visualize 3D point clouds
    Parameters:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud,where each row contains the 3D coordinates
                   of a triangulated point.
    """
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    ####################################


if __name__ == "__main__":
    data_for_fundamental_matrix = np.load("data/corresp_subset.npz")
    pts1_for_fundamental_matrix = data_for_fundamental_matrix['pts1']
    pts2_for_fundamental_matrix = data_for_fundamental_matrix['pts2']

    img1 = cv2.imread('data/im1.png')
    img2 = cv2.imread('data/im2.png')
    scale = max(img1.shape)
    F = compute_fundamental_matrix(pts1_for_fundamental_matrix,pts2_for_fundamental_matrix,scale)
    _helper.epipolar_lines_GUI_tool(img1,img2,F)
    
    data_for_temple = np.load("data/temple_coords.npz")
    pts1_epipolar = data_for_temple['pts1']

    data_for_intrinsics = np.load("data/intrinsics.npz")
    K1 = data_for_intrinsics['K1']
    K2 = data_for_intrinsics['K2']
    ####################################
    ##########YOUR CODE HERE############
    ####################################

    ####################################




