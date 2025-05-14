"""
Calculate the bounding boxes of shapes
==========

Display shapes of various types in a shapes layer. Then, calculated the bounding
box of each shape and create a new layer. Then, check if the shape is a rotated rectangle
and print the result.

.. tags:: visualization-advanced, experimental
"""

import numpy as np
from skimage import data

import napari

# add the image
viewer = napari.view_image(data.camera(), name='photographer')

# create a list of polygons
polygons = [
    np.array([[11, 13], [111, 113], [22, 246]]),
    np.array(
        [
            [505, 60],
            [402, 71],
            [383, 42],
            [251, 95],
            [212, 59],
            [131, 137],
            [126, 187],
            [191, 204],
            [171, 248],
            [211, 260],
            [273, 243],
            [264, 225],
            [430, 173],
            [512, 160],
        ]
    ),
    np.array(
        [
            [310, 382],
            [229, 381],
            [209, 401],
            [221, 411],
            [258, 411],
            [300, 412],
            [306, 435],
            [268, 434],
            [265, 454],
            [298, 461],
            [307, 461],
            [307, 507],
            [349, 510],
            [352, 369],
            [330, 366],
        ]
    ),
]

# add polygons
layer = viewer.add_shapes(
    polygons,
    shape_type='polygon',
    edge_width=1,
    edge_color='coral',
    face_color='royalblue',
    name='shapes',
)

# change some attributes of the layer
layer.selected_data = set(range(layer.nshapes))
layer.current_edge_width = 5
layer.selected_data = set()

# add an ellipse to the layer
ellipse = np.array([[59, 222], [110, 289], [170, 243], [119, 176]])
layer.add(
    ellipse,
    shape_type='ellipse',
    edge_width=5,
    edge_color='coral',
    face_color='purple',
)

# create a rectangle
rectangle = np.array([[200, 200], [300, 200], [300, 300], [200, 300]])
# add the rectangle to the viewer
layer.add(
    rectangle,
    shape_type='rectangle',
    edge_width=5,
    edge_color='coral',
    face_color='purple',
)

# create a rotated rectangle
rotated_rectangle = np.array(
    [
        [ 64.75069, 353.22742],
        [137.60655, 284.7292 ],
        [206.105  , 357.5849 ],
        [133.24908, 426.08325]
    ],
    dtype=np.float32
)
layer.add(
    rotated_rectangle,
    shape_type='rectangle',
    edge_width=5,
    edge_color='orange',
    face_color='green',
)

# get all the bounding boxes of the shapes
bboxes = []
for shape in layer._data_view.shapes:
    bboxes.append(shape._bounding_box)

    if shape.name != 'rectangle':
        continue

    verts = shape.data
    bbox = shape._bounding_box
    # Get the 4 corners of the bounding box
    bbox_corners = np.array([
        [bbox[0,0], bbox[0,1]],
        [bbox[0,0], bbox[1,1]],
        [bbox[1,0], bbox[0,1]],
        [bbox[1,0], bbox[1,1]],
    ])
    # Check if each vertex matches (within tolerance) any bbox corner
    matches = [np.any(np.all(np.isclose(v, bbox_corners), axis=1)) for v in verts]
    if all(matches):
        print('axis-aligned rectangle')
    else:
        print('rotated rectangle')


# add the bounding boxes to a new shapes layer
viewer.add_shapes(
    bboxes,
    shape_type='rectangle',
    edge_width=1,
    edge_color='yellow',
    face_color='transparent',
    name='bounding boxes',
)

if __name__ == '__main__':
    napari.run()
