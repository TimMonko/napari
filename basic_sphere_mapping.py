"""
Basic Sphere Mapping Example
============================

This example shows the fundamental approach to mapping rectangular data onto a sphere in napari.
"""

import numpy as np
from vispy.geometry import create_sphere
import napari


# Step 1: Create sample rectangular data (like a world map)
def create_sample_data():
    # Create a simple pattern that's easy to recognize on the sphere
    width, height = 180, 90  # longitude, latitude resolution
    
    # Create coordinates
    x = np.linspace(0, 2*np.pi, width)
    y = np.linspace(0, np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Create a simple pattern: stripes + circles
    data = np.sin(3*X) + np.cos(4*Y) + np.sin(np.sqrt(X**2 + Y**2))
    
    return data


# Step 2: Create sphere and map data
def main():
    # Create sample data
    rect_data = create_sample_data()
    
    # Create sphere mesh
    mesh = create_sphere(subdivisions=5, method='ico')
    vertices = mesh.get_vertices() * 100  # Scale for visibility
    faces = mesh.get_faces()
    
    # Simple mapping: use spherical coordinates
    # For each vertex, calculate its position on the rectangular array
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Convert to spherical coordinates
    # theta: azimuthal angle (longitude equivalent)
    # phi: polar angle (latitude equivalent)  
    theta = np.arctan2(y, x) + np.pi  # 0 to 2π
    phi = np.arccos(z / 100)  # 0 to π (from north pole to south pole)
    
    # Map to array indices
    height, width = rect_data.shape
    j_indices = (theta / (2*np.pi) * width).astype(int)
    i_indices = (phi / np.pi * height).astype(int)
    
    # Clamp indices to valid range
    j_indices = np.clip(j_indices, 0, width-1)
    i_indices = np.clip(i_indices, 0, height-1)
    
    # Get values for each vertex
    vertex_values = rect_data[i_indices, j_indices]
    
    # Create napari viewer
    viewer = napari.Viewer(ndisplay=3)
    
    # Add rectangular data as 2D image
    viewer.add_image(rect_data, name='Original 2D Data', colormap='viridis')
    
    # Add sphere with mapped data
    viewer.add_surface(
        (vertices, faces, vertex_values),
        name='Mapped to Sphere',
        colormap='viridis',
        translate=[0, 0, 150]  # Offset from 2D data
    )
    
    # Set camera angle
    viewer.camera.angles = (15, 25, 45)
    viewer.camera.zoom = 1.2
    
    return viewer


if __name__ == '__main__':
    viewer = main()
    napari.run()
