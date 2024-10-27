import numpy as np
import deepdrr
from deepdrr import geo, Volume
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from deepdrr import device
import os
import pydicom
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_sf_points(n):
    phi_inv = (np.sqrt(5) + 1) / 2
    points = []

    for i in range(n):
        z_i = 1 - (2 * i + 1) / n
        phi_i = 2 * np.pi * (i * phi_inv % 1) 
        theta_i = np.arccos(z_i)

        x_i = np.sin(theta_i) * np.cos(phi_i)
        y_i = np.sin(theta_i) * np.sin(phi_i)
        z_i = np.cos(theta_i)

        points.append((x_i, y_i, z_i))
    
    return points

def spherical_to_carm_angles(point):
    x, y, z = point
    r = np.sqrt(x**2 + y**2 + z**2)
    
    alpha = np.arctan2(y, x)
    beta = np.arccos(z / r)

    alpha = np.degrees(alpha)
    beta = np.degrees(beta)

    return alpha, beta
