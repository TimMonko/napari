"""
Global Weather Data on a Sphere in Napari
==========================================

This example demonstrates how to map rectangular weather data (like temperature,
precipitation, etc.) onto a 3D sphere to create a globe visualization in napari.

We'll create:
1. A sphere mesh with appropriate texture coordinates
2. Sample weather data on a lat/lon grid
3. Map the rectangular data onto the sphere
4. Visualize both the original rectangular data and the sphere projection

.. tags:: visualization-3D, weather, sphere, globe
"""

import numpy as np
from vispy.geometry import create_sphere
import napari


def create_sphere_with_texture_coords(subdivisions=4):
    """
    Create a sphere mesh with texture coordinates for mapping rectangular data.
    
    Parameters
    ----------
    subdivisions : int
        Number of subdivisions for the sphere (higher = more detailed)
        
    Returns
    -------
    vertices : ndarray
        3D coordinates of sphere vertices
    faces : ndarray
        Triangle faces connecting vertices
    texcoords : ndarray
        2D texture coordinates (u, v) for each vertex
    """
    # Create sphere using vispy
    mesh = create_sphere(subdivisions=subdivisions, method='ico')
    vertices = mesh.get_vertices()
    faces = mesh.get_faces()
    
    # Calculate texture coordinates from sphere vertices
    # Convert 3D sphere coordinates to lat/lon texture coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Calculate latitude and longitude
    lat = np.arcsin(z)  # latitude from -π/2 to π/2
    lon = np.arctan2(y, x)  # longitude from -π to π
    
    # Convert to texture coordinates (0 to 1)
    u = (lon + np.pi) / (2 * np.pi)  # longitude: 0 to 1
    v = (lat + np.pi/2) / np.pi      # latitude: 0 to 1
    
    texcoords = np.column_stack([u, v])
    
    return vertices, faces, texcoords


def generate_sample_weather_data(nlat=180, nlon=360):
    """
    Generate sample weather data on a rectangular lat/lon grid.
    This simulates global temperature data with some interesting patterns.
    
    Parameters
    ----------
    nlat : int
        Number of latitude points (typically 180 for 1-degree resolution)
    nlon : int
        Number of longitude points (typically 360 for 1-degree resolution)
        
    Returns
    -------
    weather_data : ndarray
        2D array of weather values (e.g., temperature)
    lat_grid : ndarray
        Latitude coordinates in degrees
    lon_grid : ndarray
        Longitude coordinates in degrees
    """
    # Create coordinate grids
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Generate sample temperature data with realistic patterns
    # Base temperature decreases with latitude (colder at poles)
    base_temp = 30 * np.cos(np.radians(lat_grid))
    
    # Add some longitudinal variation (continents vs oceans)
    continental_effect = 10 * np.sin(2 * np.radians(lon_grid)) * np.cos(np.radians(lat_grid))
    
    # Add some random weather patterns
    np.random.seed(42)  # For reproducibility
    weather_noise = 5 * np.random.randn(nlat, nlon)
    
    # Combine all effects
    weather_data = base_temp + continental_effect + weather_noise
    
    return weather_data, lat_grid, lon_grid


def map_data_to_sphere_vertices(weather_data, texcoords, lat_bounds=(-90, 90), lon_bounds=(-180, 180)):
    """
    Map rectangular weather data to sphere vertices using texture coordinates.
    
    Parameters
    ----------
    weather_data : ndarray
        2D array of weather data (nlat x nlon)
    texcoords : ndarray
        Texture coordinates of sphere vertices
    lat_bounds : tuple
        (min_lat, max_lat) in degrees
    lon_bounds : tuple
        (min_lon, max_lon) in degrees
        
    Returns
    -------
    vertex_values : ndarray
        Weather values interpolated to each sphere vertex
    """
    from scipy.interpolate import RegularGridInterpolator
    
    nlat, nlon = weather_data.shape
    
    # Create coordinate arrays
    lat_coords = np.linspace(lat_bounds[0], lat_bounds[1], nlat)
    lon_coords = np.linspace(lon_bounds[0], lon_bounds[1], nlon)
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (lat_coords, lon_coords), 
        weather_data, 
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Convert texture coordinates back to lat/lon
    u, v = texcoords[:, 0], texcoords[:, 1]
    lon_vertices = u * (lon_bounds[1] - lon_bounds[0]) + lon_bounds[0]
    lat_vertices = v * (lat_bounds[1] - lat_bounds[0]) + lat_bounds[0]
    
    # Interpolate weather data to vertex positions
    vertex_coords = np.column_stack([lat_vertices, lon_vertices])
    vertex_values = interpolator(vertex_coords)
    
    # Handle any NaN values (shouldn't happen with proper bounds)
    vertex_values = np.nan_to_num(vertex_values, nan=np.nanmean(vertex_values))
    
    return vertex_values


def main():
    """Create and display the weather globe visualization."""
    
    # Generate sample weather data
    print("Generating sample weather data...")
    weather_data, lat_grid, lon_grid = generate_sample_weather_data(nlat=90, nlon=180)
    
    # Create sphere mesh with texture coordinates
    print("Creating sphere mesh...")
    vertices, faces, texcoords = create_sphere_with_texture_coords(subdivisions=5)
    
    # Scale up the sphere for better visualization
    sphere_radius = 100
    vertices *= sphere_radius
    
    # Map weather data to sphere vertices
    print("Mapping data to sphere...")
    vertex_values = map_data_to_sphere_vertices(weather_data, texcoords)
    
    # Create napari viewer
    viewer = napari.Viewer(ndisplay=3)
    
    # Add the original rectangular weather data as an image
    print("Adding weather data visualization...")
    weather_layer = viewer.add_image(
        weather_data,
        name='Weather Data (2D)',
        colormap='coolwarm',
        scale=[1, 1],  # You might want to adjust this for lat/lon scaling
    )
    
    # Add the sphere with mapped weather data
    sphere_layer = viewer.add_surface(
        (vertices, faces, vertex_values),
        name='Weather Globe',
        colormap='coolwarm',
        opacity=0.9,
        shading='smooth'
    )
    
    # Position the sphere away from the 2D data for comparison
    sphere_layer.translate = [0, 0, 150]
    
    # Set up the viewer for 3D visualization
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0, 0, 90)
    viewer.camera.zoom = 1.5
    
    # Add some informative text
    print("\nVisualization created!")
    print("- The 2D image shows the original rectangular weather data")
    print("- The 3D sphere shows the same data mapped onto a globe")
    print("- Use mouse to rotate and explore the 3D view")
    print("- Both layers use the same colormap for easy comparison")
    
    return viewer


if __name__ == '__main__':
    # Check if required packages are available
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        print("Error: scipy is required for this example.")
        print("Please install scipy: pip install scipy")
        exit(1)
    
    viewer = main()
    napari.run()
