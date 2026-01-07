import numpy as np
import torch

class Camera:

    def __init__(self, image_width, image_height, focal_length, up=None):
        """
        Initialize the Camera with intrinsic parameters.

        Parameters:
            fx (float): Focal length in x (pixels).
            fy (float): Focal length in y (pixels).
            cx (float): Principal point x-coordinate (pixels).
            cy (float): Principal point y-coordinate (pixels).
        """
        self.image_width = image_width
        self.image_height = image_height
        self.up = up if up is not None else torch.tensor([0., 0., 1.])  # Default up vector
        self.focal_length = focal_length  # Assuming square pixels: fx = fy
        self.cx = image_width / 2
        self.cy = image_height / 2

    @staticmethod
    def unit_vec_from_az_el(az_deg, el_deg):
        """
        Computes a unit vector from azimuth and elevation angles (in degrees).

        Parameters:
            az_deg (float or torch.Tensor): Azimuth angle(s) in degrees.
            el_deg (float or torch.Tensor): Elevation angle(s) in degrees.

        Returns:
            torch.Tensor: Unit vector(s) corresponding to the given azimuth and elevation.
                         Shape: (3,) if scalars, (3, N) if arrays.
        """
        # Convert to tensors if needed
        if not isinstance(az_deg, torch.Tensor):
            az_deg = torch.tensor(az_deg, dtype=torch.float32)
        if not isinstance(el_deg, torch.Tensor):
            el_deg = torch.tensor(el_deg, dtype=torch.float32)
            
        az = torch.deg2rad(az_deg)  # Convert azimuth to radians
        el = torch.deg2rad(el_deg)  # Convert elevation to radians
        # Spherical to Cartesian conversion
        x = torch.sin(az) * torch.cos(el)
        y = torch.cos(az) * torch.cos(el)
        z = torch.sin(el)
        v = torch.stack([x, y, z], dim=0)
        # Normalize to unit length
        return v / torch.norm(v, dim=0)

    @staticmethod
    def look_at(camera_pos, target, up):
        """
        Computes a rotation matrix that orients the camera to look at a target point.

        Parameters:
            camera_pos (torch.Tensor): Camera position in world coordinates (3,).
            target (torch.Tensor): Target point to look at (3,).
            up (torch.Tensor): Up direction vector (3,).

        Returns:
            torch.Tensor: 3x3 rotation matrix (camera-to-world).
        """
        z = (target - camera_pos)  # Forward vector
        z = z / torch.norm(z)  # Normalize
        x = torch.linalg.cross(up, z)        # Right vector
        x = x / torch.norm(x)     # Normalize
        y = torch.linalg.cross(z, x)         # True up vector
        # Stack as rotation matrix: rows are x, y, z axes
        Rot = torch.stack([x, y, z], dim=0)
        return Rot

    @staticmethod
    def world_to_camera(X, Rot, t):
        """
        Transforms 3D points from world coordinates to camera coordinates.

        Parameters:
            X (torch.Tensor): Points in world coordinates (N, 3).
            Rot (torch.Tensor): 3x3 rotation matrix (camera-to-world).
            t (torch.Tensor): Camera position in world coordinates (3,).

        Returns:
            torch.Tensor: Points in camera coordinates (N, 3).
        """
        # Apply rotation and translation: X_cam = R @ (X - t)
        return (Rot @ (X - t).T).T

    @staticmethod
    def project_to_image(Xc, fx, fy, cx, cy):
        """
        Projects 3D camera coordinates onto a 2D image plane using pinhole camera model.

        Parameters:
            Xc (torch.Tensor): Points in camera coordinates (N, 3).
            fx (float): Focal length in x (pixels).
            fy (float): Focal length in y (pixels).
            cx (float): Principal point x-coordinate (pixels).
            cy (float): Principal point y-coordinate (pixels).

        Returns:
            torch.Tensor: 2D image coordinates (N, 2).
        """
        x, y, z = Xc[:,0], Xc[:,1], Xc[:,2]
        u = fx * (x / z) + cx  # Project x to image plane
        v = fy * (y / z) + cy  # Project y to image plane
        return torch.stack([u, v], dim=-1)
    
    def get_intrinsics(self):
        """
        Returns the intrinsic parameters of the camera.

        Returns:
            tuple: (fx, fy, cx, cy)
        """
        return (self.focal_length, self.focal_length, self.cx, self.cy)
    
   
