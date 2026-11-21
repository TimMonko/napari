"""Compare transform/unit handling between a baseline checkout and this branch.

NOT FOR COMMITTING. Run with this worktree's venv, from this worktree:

    .venv\\Scripts\\python compare_affine.py                  list scenarios
    .venv\\Scripts\\python compare_affine.py shapes-affine    two windows, side by side
    .venv\\Scripts\\python compare_affine.py --report         measured table, no interaction

Both sides use this venv; the baseline just puts another checkout's `src` first
on PYTHONPATH (default: ../napari/src, change with --baseline PATH). Each window
title says which code it is running, and the instructions are on the canvas.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENARIOS = {}


def scenario(name, summary):
    def register(cls):
        cls.name, cls.summary = name, summary
        SCENARIOS[name] = cls
        return cls

    return register


# ------------------------------------------------------------------- helpers
SQUARE = [[0, 0], [0, 100], [100, 100], [100, 0]]


def select_vertices(viewer, layer):
    viewer.layers.selection = {layer}
    viewer.layers.selection.active = layer
    layer.selected_data = {0}
    layer.mode = 'direct'


def grab_radius(viewer, layer, world_vertex):
    """Furthest canvas px from a vertex that still grabs it, for last 2 axes."""
    import numpy as np

    zoom = viewer.scene.camera.zoom
    radii = []
    for axis in viewer.dims.displayed:
        reach = 0.0
        for px in np.arange(0, 30, 0.25):
            position = np.array(world_vertex, dtype=float)
            position[axis] += px / zoom
            if layer.get_value(position, world=True)[1] is None:
                break
            reach = float(px)
        radii.append(reach)
    return '{:g} / {:g} px'.format(*radii)


def hit_diameter(viewer, layer, world_center):
    """Canvas px along each displayed axis over which a point is hit."""
    import numpy as np

    zoom = viewer.scene.camera.zoom
    sizes = []
    for axis in viewer.dims.displayed:
        hits = [
            px
            for px in np.arange(-600, 600, 2.0)
            if layer.get_value(
                np.array(world_center)
                + np.eye(len(world_center))[axis] * px / zoom,
                world=True,
            )
            is not None
        ]
        sizes.append(max(hits) - min(hits) if hits else 0)
    return '{:g} / {:g} px'.format(*sizes)


def drawn_diameter(viewer, layer):
    """Canvas px covered by a layer when it is the only one visible."""
    canvas = viewer.window._qt_viewer.canvas
    others = [x for x in viewer.layers if x is not layer]
    for other in others:
        other.visible = False
    canvas.on_draw(None)
    image = canvas._scene_canvas.render()[..., :3].astype(int)
    for other in others:
        other.visible = True
    ratio = image.shape[0] / canvas._scene_canvas.size[1]
    mask = abs(image - image[0, 0]).sum(-1) > 150
    return '{:.0f} / {:.0f} px'.format(
        mask.any(1).sum() / ratio, mask.any(0).sum() / ratio
    )


def look_at(viewer, center, zoom):
    viewer.scene.camera.center = (0, *center)
    viewer.scene.camera.zoom = zoom
    viewer.window._qt_viewer.canvas.on_draw(None)


# ----------------------------------------------------------------- scenarios
@scenario('shapes-affine', 'Vertex grab radius with scale vs affine')
class ShapesAffine:
    text = (
        "Both squares are in Direct select mode. Hover a corner, then drag it.\n"
        "Select each layer in turn. 'affine' should behave like 'scale'."
    )

    def build(self, viewer):
        import numpy as np

        self.scale = viewer.add_shapes(
            SQUARE, shape_type='rectangle', scale=(0.01, 0.01), name='scale'
        )
        matrix = np.diag([0.01, 0.01, 1.0])
        matrix[1, 2] = 1.5
        self.affine = viewer.add_shapes(
            SQUARE, shape_type='rectangle', affine=matrix, name='affine'
        )
        select_vertices(viewer, self.scale)
        select_vertices(viewer, self.affine)
        self.scale.selected_data = {0}
        self.scale.mode = 'direct'

    def report(self, viewer):
        look_at(viewer, (0.5, 1.25), 200)
        return {
            'Shapes `scale`: grab radius y / x': grab_radius(
                viewer, self.scale, (0, 0)
            ),
            'Shapes `affine`: grab radius y / x': grab_radius(
                viewer, self.affine, (0, 1.5)
            ),
        }

    expected = '5 / 5 px'


@scenario('shapes-new-layer', "Jonas's case: new Shapes layer from an affine image")
class ShapesNewLayer:
    text = (
        "New Shapes layer made from the selected affine image (values from Zulip).\n"
        'Drag a corner of the square. y is the smaller scale, so test both axes.'
    )

    def build(self, viewer):
        import numpy as np

        from napari._app_model.actions._file import new_shapes

        scale = (0.008014, 0.004171, 0.008007)
        self.image = viewer.add_image(
            np.random.default_rng(0).random((64, 256, 256)),
            affine=np.diag([*scale, 1.0]),
            name='image',
        )
        viewer.layers.selection = {self.image}
        viewer.layers.selection.active = self.image
        new_shapes(viewer)
        self.shapes = viewer.layers[-1]
        self.plane = viewer.dims.point[0] / scale[0]
        z = self.plane
        self.shapes.add_rectangles(
            [[z, 50, 50], [z, 50, 200], [z, 200, 200], [z, 200, 50]]
        )
        select_vertices(viewer, self.shapes)
        self.vertex = (viewer.dims.point[0], 50 * scale[1], 50 * scale[2])

    def report(self, viewer):
        look_at(viewer, (0.5, 1.0), 400)
        return {
            'New Shapes from affine image: grab radius y / x': grab_radius(
                viewer, self.shapes, self.vertex
            )
        }

    expected = '5 / 5 px'


@scenario('shapes-reorder', 'Vertex radius when the last data axis is hidden')
class ShapesReorder:
    text = (
        'Scale is (1, 1, 100) and dims order is (2, 0, 1), so the 100x axis is hidden.\n'
        'Drag a corner of the square.'
    )

    def build(self, viewer):
        self.shapes = viewer.add_shapes(
            [[0, 0, 5], [0, 100, 5], [100, 100, 5], [100, 0, 5]],
            shape_type='rectangle',
            scale=(1, 1, 100),
        )
        viewer.dims.order = (2, 0, 1)
        viewer.dims.point = (0, 0, 500)
        select_vertices(viewer, self.shapes)

    def report(self, viewer):
        look_at(viewer, (50, 50), 2)
        return {
            'Shapes, hidden 100x axis: grab radius': grab_radius(
                viewer, self.shapes, (0, 0, 500)
            )
        }

    expected = '5 / 5 px'


@scenario('shapes-anisotropic', 'Vertex radius with different axis scales')
class ShapesAnisotropic:
    text = (
        'Scale is (0.004, 0.008). Approach a corner along y, then along x.\n'
        'The grab distance should match.'
    )

    def build(self, viewer):
        self.shapes = viewer.add_shapes(
            SQUARE, shape_type='rectangle', scale=(0.004, 0.008)
        )
        select_vertices(viewer, self.shapes)

    def report(self, viewer):
        look_at(viewer, (0.2, 0.4), 500)
        return {
            'Shapes, anisotropic scale: grab radius y / x': grab_radius(
                viewer, self.shapes, (0, 0)
            )
        }

    expected = '5 / 5 px'


@scenario('points-affine', 'Point size and hit area with scale vs affine')
class PointsAffine:
    text = (
        "Both points have size 100 and 0.5 scale. 'affine' should match 'scale'.\n"
        'Select a layer, use Select points, and click at the edge of a point.'
    )

    def build(self, viewer):
        import numpy as np

        self.scale = viewer.add_points(
            [[100, 100]],
            size=100,
            scale=(0.5, 0.5),
            name='scale',
            face_color='white',
            border_width=0,
        )
        matrix = np.diag([0.5, 0.5, 1.0])
        matrix[1, 2] = 120
        self.affine = viewer.add_points(
            [[100, 100]],
            size=100,
            affine=matrix,
            name='affine',
            face_color='cyan',
            border_width=0,
        )
        self.scale.mode = 'select'
        self.affine.mode = 'select'

    def report(self, viewer):
        look_at(viewer, (50, 110), 2)
        return {
            'Points `scale`: drawn': drawn_diameter(viewer, self.scale),
            'Points `scale`: hit area': hit_diameter(
                viewer, self.scale, (50, 50)
            ),
            'Points `affine`: drawn': drawn_diameter(viewer, self.affine),
            'Points `affine`: hit area': hit_diameter(
                viewer, self.affine, (50, 170)
            ),
        }

    expected = '100 / 100 px'


@scenario('points-reorder', 'Point size when the last data axis is hidden')
class PointsReorder:
    text = (
        'A size-10 point with scale (1, 1, 100) and the 100x axis hidden.\n'
        'It should be 10 world units wide next to the 100-wide reference square.'
    )

    def build(self, viewer):
        viewer.add_shapes(
            [[0, 0, 5], [0, 100, 5], [100, 100, 5], [100, 0, 5]],
            shape_type='rectangle',
            scale=(1, 1, 100),
            name='100 wide reference',
            face_color='transparent',
        )
        self.points = viewer.add_points(
            [[50, 50, 5]],
            size=10,
            scale=(1, 1, 100),
            face_color='white',
            border_width=0,
        )
        viewer.dims.order = (2, 0, 1)
        viewer.dims.point = (0, 0, 500)
        self.points.mode = 'select'

    def report(self, viewer):
        look_at(viewer, (50, 50), 2)
        return {
            'Points, hidden 100x axis: drawn': drawn_diameter(
                viewer, self.points
            ),
            'Points, hidden 100x axis: hit area': hit_diameter(
                viewer, self.points, (50, 50, 500)
            ),
        }

    expected = '20 / 20 px'


@scenario('labels-brush', 'Paint brush shape with scale vs affine')
class LabelsBrush:
    text = (
        "Paint in each layer. 'affine' strokes should be as round as 'scale',\n"
        'and match the brush cursor.'
    )

    def build(self, viewer):
        import numpy as np

        self.scale = viewer.add_labels(
            np.zeros((101, 101), dtype=np.uint8), scale=(1, 4), name='scale'
        )
        matrix = np.diag([1.0, 4.0, 1.0])
        matrix[1, 2] = 450
        self.affine = viewer.add_labels(
            np.zeros((101, 101), dtype=np.uint8), affine=matrix, name='affine'
        )
        for layer in (self.scale, self.affine):
            layer.brush_size = 21
            layer.mode = 'paint'

    def report(self, viewer):
        import numpy as np

        rows = {}
        for name, layer in (('scale', self.scale), ('affine', self.affine)):
            layer.paint((50, 50), 1, refresh=False)
            ys, xs = np.nonzero(layer.data)
            rows[f'Labels `{name}`: painted area in world (y x)'] = (
                f'{(np.ptp(ys) + 1)} x {(np.ptp(xs) + 1) * 4}'
            )
        return rows

    expected = '21 x 20'


@scenario('labels-brush-drag', 'Alt-drag brush resize with a small scale')
class LabelsBrushDrag:
    text = (
        'Paint mode, scale 0.01. Hold Alt and move the mouse sideways.\n'
        'The brush edge should follow the cursor.'
    )

    def build(self, viewer):
        import numpy as np

        self.labels = viewer.add_labels(
            np.zeros((200, 200), dtype=np.uint8), scale=(0.01, 0.01)
        )
        self.labels.mode = 'paint'
        self.labels.brush_size = 10

    def report(self, viewer):
        from types import SimpleNamespace

        zoom = 200  # 2 canvas px per data px
        for x in (0, 100):
            event = SimpleNamespace(
                modifiers=('Alt',), pos=(x, 0), camera_zoom=zoom
            )
            self.labels.brush_size_on_mouse_move(self.labels, event)
        return {
            'Labels: brush size after 100 px Alt-drag (start 10, 2 px/pixel)': str(
                self.labels.brush_size
            )
        }

    expected = '60'


@scenario('labels-cursor', 'Brush cursor after the transform changes')
class LabelsCursor:
    text = (
        'Paint mode. Press j to double the scale of the layer.\n'
        'The brush cursor should grow with the layer.'
    )

    def build(self, viewer):
        import numpy as np

        self.labels = viewer.add_labels(np.zeros((200, 200), dtype=np.uint8))
        self.labels.brush_size = 10
        self.labels.mode = 'paint'

        @self.labels.bind_key('j', overwrite=True)
        def double_scale(layer):
            layer.scale = np.array(layer.scale) * 2

    def report(self, viewer):
        self.labels.scale = (2, 2)
        return {
            'Labels: cursor size after scale x2 (brush 10)': (
                f'{float(self.labels.cursor_size):g}'
            )
        }

    expected = '20'


@scenario('units', 'Layers in mm over an image in um')
class Units:
    text = (
        'The Shapes, Points and Labels layers use mm, over an image in um.\n'
        'They all should react where they are drawn. Try each layer.'
    )

    def build(self, viewer):
        import numpy as np

        viewer.add_image(
            np.random.default_rng(0).random((1000, 1000)),
            units=('um', 'um'),
            name='image (um)',
            contrast_limits=(0, 4),
        )
        self.labels = viewer.add_labels(
            np.zeros((1000, 1000), dtype=np.uint8),
            scale=(0.001, 0.001),
            units=('mm', 'mm'),
            name='labels (mm)',
        )
        self.points = viewer.add_points(
            [[0.2, 0.2]],
            size=0.1,
            units=('mm', 'mm'),
            face_color='white',
            border_width=0,
            name='points (mm)',
        )
        self.shapes = viewer.add_shapes(
            [[[0.5, 0.5], [0.5, 0.9], [0.9, 0.9], [0.9, 0.5]]],
            shape_type='rectangle',
            units=('mm', 'mm'),
            name='shapes (mm)',
        )
        self.points.mode = 'select'
        select_vertices(viewer, self.shapes)

    def report(self, viewer):
        look_at(viewer, (500, 500), 0.5)
        return {
            'Shapes in mm: grab radius at drawn vertex (500, 500)': grab_radius(
                viewer, self.shapes, (500, 500)
            ),
            'Points in mm: value at drawn point (200, 200)': str(
                self.points.get_value((200, 200), world=True)
            ),
            'Layer: data_to_world((0.2, 0.2))': str(
                tuple(
                    round(float(v), 3)
                    for v in self.points.data_to_world((0.2, 0.2))
                )
            ),
        }

    expected = '5 / 5 px; 0; (200.0, 200.0)'


@scenario('multiscale-3d', '3D multiscale position with scale vs affine')
class Multiscale3D:
    text = (
        "3D view. The 'affine' volume should sit exactly 1 world unit right of 'scale',\n"
        'not shifted further. The bounding boxes show the true extents.'
    )

    def build(self, viewer):
        import numpy as np

        data = [
            np.random.default_rng(0).random((64, 64, 64)),
            np.random.default_rng(0).random((32, 32, 32)),
        ]
        viewer.dims.ndisplay = 3
        self.scale = viewer.add_image(
            data, multiscale=True, scale=(0.01, 0.01, 0.01), name='scale'
        )
        matrix = np.diag([0.01, 0.01, 0.01, 1.0])
        matrix[2, 3] = 1.0
        self.affine = viewer.add_image(
            data, multiscale=True, affine=matrix, name='affine'
        )
        for layer in (self.scale, self.affine):
            layer.bounding_box.visible = True

    def report(self, viewer):
        import numpy as np

        canvas = viewer.window._qt_viewer.canvas
        rows = {}
        for name, layer, base in (
            ('scale', self.scale, 0.0),
            ('affine', self.affine, 1.0),
        ):
            layer._data_level = 1
            visual = canvas.layer_to_visual[layer]
            visual._on_matrix_change()
            x = visual.node.transform.matrix[-1, 0]
            rows[f'Multiscale 3D `{name}`: level 1 offset in x'] = (
                f'{float(np.round(x - base, 4)):g}'
            )
        return rows

    expected = '0.005'


@scenario('status', 'Status bar coordinates with scale vs affine')
class Status:
    text = (
        'Hover each layer and read the coordinates in the status bar.\n'
        "'affine' should show decimals like 'scale', not rounded integers."
    )

    def build(self, viewer):
        import numpy as np

        self.scale = viewer.add_image(
            np.random.default_rng(0).random((100, 100)),
            scale=(0.01, 0.01),
            name='scale',
        )
        matrix = np.diag([0.01, 0.01, 1.0])
        matrix[1, 2] = 1.5
        self.affine = viewer.add_image(
            np.random.default_rng(1).random((100, 100)),
            affine=matrix,
            name='affine',
        )

    def report(self, viewer):
        return {
            'Status `scale` at world (0.123, 0.456)': self.scale.get_status(
                (0.123, 0.456), world=True
            )['coords'],
            'Status `affine` at world (0.123, 1.956)': self.affine.get_status(
                (0.123, 1.956), world=True
            )['coords'],
        }

    expected = 'decimals, matching the position'


# -------------------------------------------------------------------- runners
def make_viewer(label, name, show, geometry=None):
    import napari

    viewer = napari.Viewer(title=f'{label}: {name}', show=show)
    canvas = viewer.window._qt_viewer.canvas
    if show and geometry:
        viewer.window._qt_window.setGeometry(*geometry)
    elif not show:
        canvas._scene_canvas.native.resize(800, 600)
    return viewer


def run_gui(name, label, geometry):
    import napari

    print(f'{label}: napari {napari.__version__} from {napari.__file__}')
    viewer = make_viewer(label, name, True, geometry)
    case = SCENARIOS[name]()
    case.build(viewer)
    overlay = viewer.canvas.overlays.text
    overlay.text = case.text
    overlay.font_size = 11
    overlay.color = 'yellow'
    overlay.visible = True
    napari.run()


def run_report():
    """Print one JSON object per scenario to stdout."""
    import warnings

    import napari

    warnings.simplefilter('ignore')
    print('NAPARI ' + napari.__file__, flush=True)
    for name, cls in SCENARIOS.items():
        viewer = make_viewer('report', name, False)
        case = cls()
        try:
            case.build(viewer)
            rows = case.report(viewer)
        except Exception as e:  # noqa: BLE001
            rows = {f'{name}: ERROR': f'{type(e).__name__}: {e}'}
        print(
            'ROW '
            + json.dumps({'name': name, 'expected': cls.expected, 'rows': rows}),
            flush=True,
        )
        viewer.close()


def child_env(src):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(src)
    return env


def commit_of(src):
    result = subprocess.run(
        ['git', '-C', str(src), 'log', '-1', '--format=%h %s'],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()[:70]


def spawn_args(*extra):
    return [sys.executable, str(Path(__file__).resolve()), *extra]


def collect(src):
    result = subprocess.run(
        spawn_args('--child-report'),
        env=child_env(src),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if result.returncode:
        print(result.stderr[-2000:])
    origin, rows, expected = None, {}, {}
    for line in result.stdout.splitlines():
        if line.startswith('NAPARI '):
            origin = line[7:]
        if line.startswith('ROW '):
            data = json.loads(line[4:])
            for check, value in data['rows'].items():
                rows[check] = value
                expected[check] = data['expected']
    return origin, rows, expected


def report(baseline):
    branch = HERE / 'src'
    base_origin, base_rows, _ = collect(baseline)
    branch_origin, branch_rows, expected = collect(branch)
    print(f'baseline: {commit_of(baseline)}\n  imported {base_origin}')
    print(f'branch:   {commit_of(branch)}\n  imported {branch_origin}\n')
    headers = ('check', 'baseline', 'branch', 'expected')
    table = [
        (
            check,
            str(base_rows.get(check, '-')),
            str(branch_rows.get(check, '-')),
            expected.get(check, ''),
        )
        for check in {**base_rows, **branch_rows}
    ]
    widths = [
        max(len(row[i]) for row in [headers, *table])
        for i in range(len(headers))
    ]
    for row in [headers, tuple('-' * w for w in widths), *table]:
        print(' | '.join(c.ljust(w) for c, w in zip(row, widths)).rstrip())


def launch_pair(name, baseline):
    # left half and right half of a 1920 wide screen
    sides = [
        ('BASELINE', baseline, '60,60,900,800'),
        ('BRANCH', HERE / 'src', '980,60,900,800'),
    ]
    print(f'Scenario {name}: {SCENARIOS[name].summary}')
    print(SCENARIOS[name].text)
    processes = []
    for label, src, geometry in sides:
        print(f'{label}: {commit_of(src)}')
        processes.append(
            subprocess.Popen(
                spawn_args(
                    name, '--child', label, '--geometry', geometry
                ),
                env=child_env(src),
            )
        )
    for process in processes:
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('scenario', nargs='?', choices=[*SCENARIOS, None])
    parser.add_argument('--report', action='store_true')
    parser.add_argument(
        '--baseline',
        type=Path,
        default=HERE.parent / 'napari' / 'src',
        help='src directory of the checkout to compare against',
    )
    parser.add_argument('--child', help=argparse.SUPPRESS)
    parser.add_argument('--child-report', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--geometry', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_report:
        run_report()
    elif args.child:
        geometry = [int(v) for v in args.geometry.split(',')]
        run_gui(args.scenario, args.child, geometry)
    elif args.report:
        report(args.baseline)
    elif args.scenario:
        launch_pair(args.scenario, args.baseline)
    else:
        print('Scenarios:')
        for name, cls in SCENARIOS.items():
            print(f'  {name:20s} {cls.summary}')
        print('\nRun one for side-by-side windows, or --report for a table.')


if __name__ == '__main__':
    main()
