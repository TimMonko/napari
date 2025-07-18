"""
Simple Weather Globe Example for Napari
=======================================

A minimal example showing how to map rectangular array data onto a sphere.
This demonstrates the basic concept of transforming a 2D array (like a world map)
onto a 3D globe visualization.
"""

import numpy as np
from vispy.geometry import create_sphere
import napari


def create_simple_weather_globe():
    """Create a simple example of mapping 2D data onto a 3D sphere."""
    
    # 1. Create sample rectangular data (like a world map)
    # This represents some weather data on a lat/lon grid
    nlat, nlon = 90, 180  # Simplified resolution
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Create sample "temperature" data with interesting patterns
    temperature_data = (
        20 * np.cos(np.radians(lat_grid)) +  # Temperature decreases toward poles
        5 * np.sin(2 * np.radians(lon_grid)) * np.cos(np.radians(lat_grid)) +  # Continental effects
        3 * np.random.randn(nlat, nlon)  # Some noise
    )
    
    # 2. Create a sphere mesh
    mesh = create_sphere(subdivisions=4, method='ico')
    vertices = mesh.get_vertices() * 50  # Scale up for visibility
    faces = mesh.get_faces()
    
    # 3. Map the rectangular data to sphere vertices
    # Convert sphere vertices to lat/lon coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Calculate lat/lon from sphere coordinates
    lat_vertices = np.degrees(np.arcsin(z / 50))  # Convert back to lat
    lon_vertices = np.degrees(np.arctan2(y, x))   # Convert back to lon
    
    # Interpolate temperature data to sphere vertices
    # Simple nearest neighbor approach
    vertex_temperatures = []
    for lat_v, lon_v in zip(lat_vertices, lon_vertices):
        # Find closest grid point
        lat_idx = np.argmin(np.abs(lat - lat_v))
        lon_idx = np.argmin(np.abs(lon - lon_v))
        vertex_temperatures.append(temperature_data[lat_idx, lon_idx])
    
    vertex_temperatures = np.array(vertex_temperatures)
    
    # 4. Create napari visualization
    viewer = napari.Viewer(ndisplay=3)
    
    # Add original 2D data
    viewer.add_image(
        temperature_data,
        name='2D Weather Data',
        colormap='coolwarm'
    )
    
    # Add 3D sphere with mapped data
    viewer.add_surface(
        (vertices, faces, vertex_temperatures),
        name='Weather Globe',
        colormap='coolwarm',
        translate=[0, 0, 100]  # Move sphere away from 2D data
    )
    
    # Set camera for good 3D view
    viewer.camera.angles = (0, 0, 90)
    viewer.camera.zoom = 2
    
    return viewer


if __name__ == '__main__':
    np.random.seed(42)  # For reproducible results
    viewer = create_simple_weather_globe()
    napari.run()
