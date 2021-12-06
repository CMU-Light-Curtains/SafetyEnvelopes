import matplotlib.pyplot as plt
import numpy as np


class Box:
    """
    A "Box" represents a rectangular object in the scene.
    It consists of 4 vertices, that are appropriately positioned and rotated.
    """

    def __init__(self,
                 x: float,
                 z: float,
                 w: float,
                 h: float,
                 rot: float,
                 translate: str):
        """
        Box uses the camera coordinate frame.

        Args:
            x (float): x coordinate of the box's center
            z (float): z coordinate of the box's center
            w (float): box width
            h (float): box heigh
            rot (float): angle of rotation, in degrees, in the counter-clockwise direction
            translate (str): can either be "center" or "near_edge" whether to translate the center of the box, or the
                             the center of the near edge
        """
        self.x = x
        self.z = z
        self.w = w
        self.h = h
        self.rot = rot
        self.hit_info = None

        # enumerate vertices in the local coordinates (box center is the origin)
        # order of enumeration is in the counter-clockwise direction starting from the lower left vertex.
        self.vertices = np.array([[-w/2, -h/2],
                                  [+w/2, -h/2],
                                  [+w/2, +h/2],
                                  [-w/2, +h/2]], dtype=np.float32)

        # rotation
        rot_rad = np.deg2rad(self.rot)
        r_matrix = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                             [np.sin(rot_rad),  np.cos(rot_rad)]], dtype=np.float32)
        self.vertices = self.vertices @ r_matrix.T

        # translation
        if translate == 'center':
            t_vector = np.array([[self.x, self.z]], dtype=np.float32)
        elif translate == 'near_edge':
            edges = self.surface()
            edge_centers = 0.5 * (edges[:, :2] + edges[:, 2:])  # (4, 2)
            near_edge_center = edge_centers[np.argmin(edge_centers[:, 1])]  # (2,)
            t_vector = np.array([[self.x, self.z]], dtype=np.float32) - near_edge_center
        else:
            raise Exception("translate must be either <center> or <near_edge>")

        # transform vertices using rotation and translation
        self.vertices = self.vertices + t_vector

    def area(self):
        return self.w * self.h

    def surface(self):
        """
        Returns:
            surface (np.ndarray, dtype=float32, shape=(4, 4)): each row corresponds to an edge of the box, in the
                                                               format (x1, z1, x2, z2).
        """
        # start and end vertices for the 4 edges
        v1 = self.vertices[[0, 1, 2, 3]]
        v2 = self.vertices[[1, 2, 3, 0]]
        return np.hstack([v1, v2])  # (4, 4)

    def draw(self):
        surface = self.surface()
        for edge in surface:
            x1, z1, x2, z2 = edge
            plt.plot([x1, x2], [z1, z2], c='b', linewidth=2)

    def is_inside_range(self, max_range):
        ranges = np.linalg.norm(self.vertices, axis=1)  # (4,)
        return np.all(ranges <= max_range)


if __name__ == '__main__':
    b1 = Box(x=4, z=4, w=2, h=2, rot=0)
    b2 = Box(x=1, z=2, w=3, h=3, rot=45)
    b3 = Box(x=-1, z=-1, w=2, h=1, rot=90)

    for b in (b1, b2, b3):
        b.draw()

    plt.xlim([-3.5, 5.5])
    plt.ylim([-3, 6])
    plt.show()

    print(f"b3 is inside 2: {b3.is_inside_range(2)}")
    print(f"b3 is inside 3: {b3.is_inside_range(3)}")
