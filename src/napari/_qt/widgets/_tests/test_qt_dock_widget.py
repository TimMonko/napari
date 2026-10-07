import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QPushButton,
    QTabBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari._qt.utils import combine_widgets


def test_add_dock_widget(make_napari_viewer):
    """Test basic add_dock_widget functionality"""
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    dwidg = viewer.window.add_dock_widget(widg, name='test', area='bottom')
    assert not dwidg.is_vertical
    assert viewer.window._qt_window.findChild(QDockWidget, 'test')
    assert dwidg.widget() == widg
    assert dwidg.titleBarWidget() is dwidg.title
    assert dwidg.title.vertical
    assert viewer.window.dock_widgets['test'] is widg
    with pytest.raises(KeyError):
        assert viewer.window.dock_widgets['test2']

    widg2 = QPushButton('button')
    dwidg2 = viewer.window.add_dock_widget(widg2, name='test2', area='right')
    assert dwidg2.is_vertical
    assert viewer.window._qt_window.findChild(QDockWidget, 'test2')
    assert dwidg2.widget() == widg2
    assert dwidg2.titleBarWidget() is dwidg2.title
    assert not dwidg2.title.vertical

    with pytest.raises(ValueError, match='area argument must be'):
        viewer.window.add_dock_widget(widg2, name='test2', area='under')

    with pytest.raises(ValueError, match='all allowed_areas argument must be'):
        viewer.window.add_dock_widget(
            widg2, name='test2', allowed_areas=['under']
        )

    with pytest.raises(TypeError, match='`allowed_areas` must be a list'):
        viewer.window.add_dock_widget(
            widg2, name='test2', allowed_areas='under'
        )


def test_add_dock_widget_from_list(make_napari_viewer):
    """Test that we can add a list of widgets and they will be combined"""
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    widg2 = QPushButton('button')

    dwidg = viewer.window.add_dock_widget(
        [widg, widg2], name='test', area='right'
    )
    assert viewer.window._qt_window.findChild(QDockWidget, 'test')
    assert isinstance(dwidg.widget().layout(), QVBoxLayout)

    dwidg = viewer.window.add_dock_widget(
        [widg, widg2], name='test2', area='bottom'
    )
    assert viewer.window._qt_window.findChild(QDockWidget, 'test2')
    assert isinstance(dwidg.widget().layout(), QHBoxLayout)


def test_add_dock_widget_raises(make_napari_viewer):
    """Test that the widget passed must be a DockWidget."""
    viewer = make_napari_viewer()
    widg = object()

    with pytest.raises(TypeError):
        viewer.window.add_dock_widget(widg, name='test')


def test_remove_dock_widget_orphans_widget(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    qtbot.addWidget(widg)

    assert not widg.parent()
    dw = viewer.window.add_dock_widget(
        widg, name='test', menu=viewer.window.window_menu
    )
    assert widg.parent() is dw
    assert dw.toggleViewAction() in viewer.window.window_menu.actions()
    viewer.window.remove_dock_widget(dw, menu=viewer.window.window_menu)
    assert dw.toggleViewAction() not in viewer.window.window_menu.actions()
    del dw
    # if dw didn't release widg, we'd get an exception when next accessing widg
    assert not widg.parent()


def test_remove_dock_widget_by_widget_reference(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    qtbot.addWidget(widg)

    dw = viewer.window.add_dock_widget(widg, name='test')
    assert widg.parent() is dw
    assert dw in viewer.window._qt_window.findChildren(QDockWidget)
    viewer.window.remove_dock_widget(widg)
    with pytest.raises(LookupError):
        # it's gone this time:
        viewer.window.remove_dock_widget(widg)
    assert not widg.parent()


def test_adding_modified_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widg = QWidget()
    # not uncommon to see people shadow the builtin layout()
    # which breaks our ability to add vertical stretch... but shouldn't crash
    widg.layout = None
    dw = viewer.window.add_dock_widget(widg, name='test', area='right')
    assert dw.widget() is widg


def test_adding_stretch(make_napari_viewer):
    """Make sure that vertical stretch only gets added when appropriate."""
    viewer = make_napari_viewer()

    # adding a widget to the left/right will usually addStretch to the layout
    widg = QWidget()
    widg.setLayout(QVBoxLayout())
    widg.layout().addWidget(QPushButton())
    assert widg.layout().count() == 1
    dw = viewer.window.add_dock_widget(widg, area='right')
    assert widg.layout().count() == 2
    dw.close()

    # ... unless the widget has a widget with a large vertical sizePolicy
    widg = QWidget()
    widg.setLayout(QVBoxLayout())
    widg.layout().addWidget(QTextEdit())
    assert widg.layout().count() == 1
    dw = viewer.window.add_dock_widget(widg, area='right')
    assert widg.layout().count() == 1
    dw.close()

    # ... widgets on the bottom do not get stretch
    widg = QWidget()
    widg.setLayout(QHBoxLayout())
    widg.layout().addWidget(QPushButton())
    assert widg.layout().count() == 1
    dw = viewer.window.add_dock_widget(widg, area='bottom')
    assert widg.layout().count() == 1
    dw.close()


def test_float_unfloat_title_bar(make_napari_viewer):
    """Float then unfloat must preserve the custom title bar with correct orientation.

    Regression test for https://github.com/napari/napari/issues/8887.

    The custom title bar must always be set (docked and floating) so that
    napari's dark theming and custom controls are always visible.

    Orientation rules:
      - docked left/right (is_vertical=True)  → vertical=False (horizontal bar)
      - docked top/bottom (is_vertical=False) → vertical=True  (side bar)
      - floating                               → always vertical=False
    """
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    dw = viewer.window.add_dock_widget(widg, name='test', area='right')

    assert dw.titleBarWidget() is dw.title
    assert not dw.title.vertical  # right-docked -> horizontal bar

    dw.setFloating(True)
    assert dw.isFloating()
    assert dw.titleBarWidget() is dw.title
    # Floating always uses a horizontal title bar regardless of window size.
    assert not dw.title.vertical

    # Re-dock the widget.
    dw.setFloating(False)
    assert not dw.isFloating()
    # Custom title bar restored with correct orientation.
    assert dw.titleBarWidget() is dw.title


def test_float_from_bottom_clears_vertical_titlebar_feature(
    make_napari_viewer,
):
    """Floating a widget from top/bottom must not leave the title bar as a left-side strip.

    Regression test for https://github.com/napari/napari/issues/8887.

    When docked at top/bottom Qt sets DockWidgetVerticalTitleBar, placing the
    custom title bar on the left side of the widget.  If that feature is not
    cleared when the widget becomes floating, the title bar ends up as a
    narrow unusable strip on the left edge of the floating window.
    """
    from qtpy.QtWidgets import QDockWidget

    viewer = make_napari_viewer()
    widg = QPushButton('button')
    dw = viewer.window.add_dock_widget(widg, name='test', area='bottom')

    # Docked at bottom: vertical feature should be set (sidebar-style bar).
    assert (
        dw.features()
        & QDockWidget.DockWidgetFeature.DockWidgetVerticalTitleBar
    )

    # Float the widget.
    dw.setFloating(True)
    assert dw.isFloating()
    # DockWidgetVerticalTitleBar must be cleared so the title bar spans the
    # top of the floating window, not the left edge.
    assert not (
        dw.features()
        & QDockWidget.DockWidgetFeature.DockWidgetVerticalTitleBar
    )
    assert dw.titleBarWidget() is dw.title
    assert not dw.title.vertical  # horizontal bar on the floating window


def test_dock_area_change_updates_custom_title_bar(make_napari_viewer):
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    dw = viewer.window.add_dock_widget(widg, name='test', area='right')

    assert not dw.title.vertical
    assert not (
        dw.features()
        & QDockWidget.DockWidgetFeature.DockWidgetVerticalTitleBar
    )

    viewer.window._qt_window.addDockWidget(
        Qt.DockWidgetArea.BottomDockWidgetArea, dw
    )

    assert dw.titleBarWidget() is dw.title
    assert dw.title.vertical
    assert (
        dw.features()
        & QDockWidget.DockWidgetFeature.DockWidgetVerticalTitleBar
    )


def test_combine_widgets_error():
    """Check error raised when combining widgets with invalid types."""
    with pytest.raises(
        TypeError, match='"widgets" must be a QWidget, a magicgui'
    ):
        combine_widgets(['string'])


def _tall_widget(height):
    widg = QWidget()
    widg.setLayout(QVBoxLayout())
    spacer = QWidget()
    spacer.setFixedHeight(height)
    widg.layout().addWidget(spacer)
    return widg


@pytest.fixture
def fake_screen(monkeypatch):
    """Pretend the screen has a fixed available height."""
    from types import SimpleNamespace

    from qtpy.QtCore import QRect

    from napari._qt.qt_main_window import _QtMainWindow

    def _set(height):
        screen = SimpleNamespace(
            availableGeometry=lambda: QRect(0, 0, 3000, height)
        )
        monkeypatch.setattr(_QtMainWindow, 'screen', lambda self: screen)

    return _set


def _add_docks(viewer, qtbot, widgets, area='right'):
    docks = []
    for i, widg in enumerate(widgets):
        qtbot.addWidget(widg)
        docks.append(
            viewer.window.add_dock_widget(widg, name=f'w{i}', area=area)
        )
        qtbot.wait(50)
    qtbot.wait(100)
    return docks


def _stacked_groups(qt_window, docks):
    groups = {
        frozenset([dw, *qt_window.tabifiedDockWidgets(dw)]) for dw in docks
    }
    return len(groups)


def test_plugin_dock_is_scrollable_but_transparent(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widg = QPushButton('button')
    qtbot.addWidget(widg)
    dw = viewer.window.add_dock_widget(widg, name='test', area='right')

    assert dw._scroll_area is not None
    assert dw.widget() is widg
    assert widg.parent() is dw
    assert viewer.window.dock_widgets['test'] is widg
    # small widgets aren't squished: the dock can't shrink below them
    assert dw.minimumSizeHint().height() >= widg.minimumSizeHint().height()

    # napari's own docks are not wrapped
    assert viewer.window._qt_viewer.dockLayerList._scroll_area is None


def test_tall_dock_widget_grows_window(make_napari_viewer, fake_screen, qtbot):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    fake_screen(10_000)
    height = qt_window.height()

    _add_docks(viewer, qtbot, [_tall_widget(height + 400)])
    qtbot.waitUntil(lambda: qt_window.height() > height + 300)


def test_short_dock_widget_keeps_window_height(
    make_napari_viewer, fake_screen, qtbot
):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    fake_screen(10_000)
    height = qt_window.height()

    _add_docks(viewer, qtbot, [_tall_widget(20)])
    assert qt_window.height() == height


def test_dock_widget_too_tall_for_screen_scrolls(
    make_napari_viewer, fake_screen, qtbot
):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    height = qt_window.height()
    fake_screen(height + 100)

    widg = _tall_widget(height * 3)
    qtbot.addWidget(widg)
    dw = viewer.window.add_dock_widget(widg, name='tall', area='right')
    qtbot.wait(200)

    # the window never grows past the screen; the dock scrolls instead
    assert qt_window.height() <= height + 100
    assert dw.height() < widg.minimumSizeHint().height()
    assert dw._scroll_area.verticalScrollBar().maximum() > 0
    # lookups through the wrapper still behave
    assert viewer.window.dock_widgets['tall'] is widg
    assert widg.parent() is dw
    assert dw.inner_widget() is widg

    viewer.window.remove_dock_widget(widg)
    assert not widg.parent()
    assert 'tall' not in viewer.window.dock_widgets


def test_tall_docks_get_a_quarter_of_the_area_before_tabifying(
    make_napari_viewer, fake_screen, qtbot
):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    fake_screen(1200)

    docks = _add_docks(viewer, qtbot, [_tall_widget(800) for _ in range(6)])
    assert _stacked_groups(qt_window, docks) == 4
    assert qt_window.height() <= 1200
    # (Qt moves background tabs off-screen rather than hiding them)
    shown = [dw for dw in docks if not dw.visibleRegion().isEmpty()]
    assert len(shown) == 4
    # roughly a quarter of the area each, comfortably above the 200px floor
    qtbot.waitUntil(lambda: all(dw.height() > 230 for dw in shown))


def test_tabs_are_added_in_the_order_the_docks_were_added(
    make_napari_viewer, fake_screen, qtbot
):
    """Tabs keep to one order, so a new tab isn't inserted at random.

    Qt reparents the members of a tab group, so reading a group back gives
    its docks in no particular order: which one ``_dock_groups`` saw first
    decided where the next tab went, and that changed between runs.
    """
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    fake_screen(1200)

    _add_docks(viewer, qtbot, [_tall_widget(800) for _ in range(6)])
    area = Qt.DockWidgetArea.RightDockWidgetArea
    groups = qt_window._dock_groups(area, ())

    assert [[dw.objectName() for dw in group] for group in groups] == [
        ['w0'],
        ['w1'],
        ['w2'],
        ['w3', 'w4', 'w5'],
    ]
    (tab_bar,) = qt_window.findChildren(QTabBar)
    assert [tab_bar.tabText(i) for i in range(tab_bar.count())] == [
        'w3',
        'w4',
        'w5',
    ]


def test_closing_a_dock_gives_the_window_height_back(
    make_napari_viewer, fake_screen, qtbot
):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    fake_screen(10_000)
    height = qt_window.height()

    (dw,) = _add_docks(viewer, qtbot, [_tall_widget(height + 400)])
    qtbot.waitUntil(lambda: qt_window.height() > height)

    viewer.window.remove_dock_widget(dw)
    qtbot.waitUntil(lambda: qt_window.height() == height)


def test_new_dock_gets_more_than_its_minimum(
    make_napari_viewer, fake_screen, qtbot
):
    viewer = make_napari_viewer(show=True)
    fake_screen(1200)

    *old, new = _add_docks(
        viewer, qtbot, [_tall_widget(800) for _ in range(3)]
    )
    qtbot.waitUntil(lambda: all(new.height() > dw.height() for dw in old))


def test_small_docks_keep_their_height_in_an_overflowing_area(
    make_napari_viewer, fake_screen, qtbot
):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    fake_screen(1200)

    widgets = [_tall_widget(800) for _ in range(3)]
    widgets += [QPushButton(f'button {i}') for i in range(2)]
    docks = _add_docks(viewer, qtbot, widgets)
    small = docks[3:]

    # small docks aren't squished below their real height...
    assert all(
        dw.height() >= dw.widget().minimumSizeHint().height() for dw in small
    )
    # ...and only take that much room, so all five fit stacked
    assert _stacked_groups(qt_window, docks) == 5


def test_docks_can_still_move_between_areas(make_napari_viewer, qtbot):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window
    widg = _tall_widget(300)
    (dw,) = _add_docks(viewer, qtbot, [widg])

    qt_window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dw)
    qtbot.wait(100)
    assert qt_window.dockWidgetArea(dw) == Qt.DockWidgetArea.LeftDockWidgetArea
    assert widg.parent() is dw
    assert dw.widget() is widg


@pytest.mark.parametrize(
    ('minimums', 'wants', 'priority', 'space', 'expected'),
    [
        # everyone gets their minimum, the new dock grows first
        (
            [100, 100, 100],
            [500, 500, 500],
            [False, False, True],
            500,
            [100, 100, 300],
        ),
        # leftovers go to the others, by how much more they want
        (
            [100, 100, 100],
            [300, 200, 150],
            [False, False, True],
            500,
            [200, 150, 150],
        ),
        # not even the minimums fit: minimums are kept
        ([200, 200], [400, 400], [True, False], 300, [200, 200]),
    ],
)
def test_share_space(minimums, wants, priority, space, expected):
    from napari._qt.qt_main_window import _share_space

    assert _share_space(minimums, wants, priority, space) == expected
