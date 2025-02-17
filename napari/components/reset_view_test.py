if __name__ == '__main__':
    import napari

    viewer = napari.Viewer()
    viewer.open_sample('napari', 'cells3d')
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (50, 12, 70)
    viewer.reset_view(reset_camera_angle=False)

    napari.run()
