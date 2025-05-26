import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=200):
        
        self.R = R
        self.w = w
        self.n = n

        # Create evenly spaced values for parameters u and v
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)

        # Create 2D grids from u and v values
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Generate the 3D mesh points
        self.X, self.Y, self.Z = self._generate_mesh()

    def _generate_mesh(self):
        
        U, V = self.U, self.V
        R = self.R

        X = (R + V * np.cos(U / 2)) * np.cos(U)
        Y = (R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)

        return X, Y, Z

    def surface_area(self):
        
        
        dXdu, dXdv = np.gradient(self.X, axis=(1, 0))
        dYdu, dYdv = np.gradient(self.Y, axis=(1, 0))
        dZdu, dZdv = np.gradient(self.Z, axis=(1, 0))

       
        cross = np.cross(
            np.stack((dXdu, dYdu, dZdu), axis=-1),
            np.stack((dXdv, dYdv, dZdv), axis=-1)
        )

    
        dA = np.linalg.norm(cross, axis=-1)

       
        area = simpson(simpson(dA, self.v), self.u)
        return area

    def edge_length(self):
        
        u = self.u
        v_edges = [-self.w / 2, self.w / 2] 
        length = 0

        for v in v_edges:
            x = (self.R + v * np.cos(u / 2)) * np.cos(u)
            y = (self.R + v * np.cos(u / 2)) * np.sin(u)
            z = v * np.sin(u / 2)
            
         
            pts = np.stack((x, y, z), axis=1)
            seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            length += np.sum(seg_lengths)

        return length

    def plot(self):
       
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, color='cyan', edgecolor='k', alpha=0.8)
        ax.set_title("3D Visualization of Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

# Run the demo
if __name__ == "__main__":
    strip = MobiusStrip(R=1.0, w=0.2, n=200)
    area = strip.surface_area()
    length = strip.edge_length()

    print(f"Estimated Surface Area: {area:.5f} units²")
    print(f"Estimated Edge Length: {length:.5f} units")

    strip.plot()
