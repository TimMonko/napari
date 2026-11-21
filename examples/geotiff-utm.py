"""
Geospatial raster in UTM coordinates
====================================

This example opens a satellite image stored as a GeoTIFF in a UTM projection
and shows it in its world coordinates, in metres.

A GeoTIFF's affine transform maps pixel indices to world coordinates, so it
translates directly into a layer's ``scale`` and ``translate``. The
coordinate grid, scale bar and axes then show positions in those world
coordinates and their units.

Requires rasterio to be installed.

.. tags:: visualization-advanced, geodata
"""

import pooch
import rasterio

import napari

# an RGB Landsat scene in UTM zone 18N, from the rasterio test data
path = pooch.retrieve(
    'https://github.com/rasterio/rasterio/raw/1.4.3/tests/data/RGB.byte.tif',
    known_hash='sha256:d7cbe932c7ed74a627706a9e9df99f706df3e5abc7d45a49e9d00677a6b09eb4',
)

with rasterio.open(path) as src:
    rgb = src.read().transpose(1, 2, 0)  # (band, row, col) -> RGB
    transform = src.transform

# the transform maps (col, row) to (easting, northing). It points to the
# corner of the first pixel, but napari places pixel centers on the grid,
# so translate to the center of the first pixel instead
x0, y0 = transform * (0.5, 0.5)

# northing decreases down the rows, so the row scale is negative
viewer, layer = napari.imshow(
    rgb,
    rgb=True,
    scale=(transform.e, transform.a),
    translate=(y0, x0),
    axis_labels=('northing', 'easting'),
    units=('m', 'm'),
)

# northing increases northwards, so point the camera's vertical axis up
# to put north at the top
viewer.scene.camera.orientation2d = ('up', 'right')
viewer.scene.overlays.coordinate_grid.visible = True
viewer.canvas.overlays.scale_bar.visible = True
viewer.canvas.overlays.axes.visible = True
viewer.fit_to_view()

# To also view the scene in latitude/longitude (EPSG:4326), rasterio can
# reproject it while reading through a WarpedVRT: a virtual raster that warps
# the pixels into the new coordinate system on the fly. Its transform is in
# degrees, so the layer needs latitude/longitude axis labels and degree
# units. Degrees and metres can't be converted into each other, so it also
# needs its own viewer rather than another layer in this one.

from rasterio.vrt import WarpedVRT

with rasterio.open(path) as src, WarpedVRT(src, crs='EPSG:4326') as vrt:
    rgb = vrt.read().transpose(1, 2, 0)
    transform = vrt.transform
x0, y0 = transform * (0.5, 0.5)
latlon_viewer, _ = napari.imshow(
    rgb,
    rgb=True,
    scale=(transform.e, transform.a),
    translate=(y0, x0),
    axis_labels=('latitude', 'longitude'),
    units=('degree', 'degree'),
)
latlon_viewer.scene.camera.orientation2d = ('up', 'right')
latlon_viewer.scene.overlays.coordinate_grid.visible = True

if __name__ == '__main__':
    napari.run()
