import sys

import pytest

from napari.layers import Points
from napari.utils._test_utils import MouseEvent, read_only_mouse_event
from napari.utils.interactions import (
    Shortcut,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


@pytest.mark.parametrize(
    ('shortcut', 'reason'),
    [
        ('Atl-A', 'Alt misspelled'),
        ('Ctrl-AA', 'AA makes no sense'),
        ('BB', 'BB makes no sense'),
    ],
)
def test_shortcut_invalid(shortcut, reason):
    with pytest.warns(UserWarning, match='does not seem to be a valid'):
        Shortcut(shortcut)  # Should be Control-A


def test_minus_shortcut():
    """
    Misc tests minus is properly handled as it is the delimiter
    """
    assert str(Shortcut('-')) == '-'
    assert str(Shortcut('Control--')).endswith('-')
    assert str(Shortcut('Shift--')).endswith('-')


def test_shortcut_qt():
    assert Shortcut('Control-A').qt == 'Ctrl+A'


@pytest.mark.skipif(
    sys.platform != 'darwin', reason='Parsing macos specific keys'
)
@pytest.mark.parametrize(
    ('expected', 'shortcut'),
    [
        ('␣', 'Space'),
        ('⌥', 'Alt'),
        ('⌥-', 'Alt--'),
        ('⌘', 'Meta'),
        ('⌘-', 'Meta--'),
        ('⌘⌥', 'Meta-Alt'),
        ('⌥⌘P', 'Meta-Alt-P'),
    ],
)
def test_partial_shortcuts(shortcut, expected):
    assert str(Shortcut(shortcut)) == expected


def _drag_logger(log):
    def drag(layer, event):
        log.append('press')
        yield
        while event.type == 'mouse_move':
            log.append('move')
            yield
        log.append('release')

    return drag


def test_generator_exception_warns_and_is_cleaned_up():
    layer = Points()

    def drag(layer, event):
        yield
        raise ValueError('boom')

    layer.mouse_drag_callbacks.append(drag)
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    with pytest.warns(UserWarning, match=r'(?s)Traceback.*boom'):
        mouse_move_callbacks(
            layer, read_only_mouse_event(type='mouse_move', is_dragging=True)
        )
    assert not layer._mouse_drag_gen
    assert not layer._persisted_mouse_event


def test_release_cleans_up_when_generator_raises():
    layer = Points()

    def drag(layer, event):
        yield
        while event.type == 'mouse_move':
            yield
        raise ValueError('boom on release')

    layer.mouse_drag_callbacks.append(drag)
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    with pytest.warns(UserWarning, match='boom on release'):
        mouse_release_callbacks(
            layer, read_only_mouse_event(type='mouse_release')
        )
    assert not layer._mouse_drag_gen
    assert not layer._persisted_mouse_event

    # the next press starts a new drag
    log = []
    layer.mouse_drag_callbacks[:] = [_drag_logger(log)]
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    assert log == ['press']


def test_active_generator_accepts_unwrapped_events():
    """Callers (e.g. plugin tests) may pass plain, non-wrapped events."""
    layer = Points()
    log = []
    layer.mouse_drag_callbacks.append(_drag_logger(log))

    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    mouse_move_callbacks(
        layer, MouseEvent(type='mouse_move', is_dragging=True)
    )
    mouse_release_callbacks(layer, MouseEvent(type='mouse_release'))

    assert log == ['press', 'move', 'release']


def test_failing_close_does_not_leave_generators_registered():
    layer = Points()

    def drag(layer, event):
        try:
            while True:
                yield
        finally:
            raise ValueError('boom in finally')

    layer.mouse_drag_callbacks.append(drag)
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    with pytest.warns(UserWarning, match='boom in finally'):
        mouse_release_callbacks(
            layer, read_only_mouse_event(type='mouse_release')
        )
    assert not layer._mouse_drag_gen
    assert not layer._persisted_mouse_event


def _dragging(event_type, button, press):
    return read_only_mouse_event(
        type=event_type, button=button, is_dragging=True, press_event=press
    )


def test_second_button_does_not_end_drag():
    """Like vispy, a drag lasts until its own button is released."""
    layer = Points()
    log = []

    def drag(layer, event):
        log.append('press')
        yield
        while event.is_dragging:
            log.append(event.type)
            yield
        log.append('done')

    layer.mouse_drag_callbacks.append(drag)
    press = MouseEvent(type='mouse_press', button=1)
    mouse_press_callbacks(layer, read_only_mouse_event(**vars(press)))
    mouse_press_callbacks(layer, _dragging('mouse_press', 2, press))
    mouse_release_callbacks(layer, _dragging('mouse_release', 2, press))
    mouse_move_callbacks(layer, _dragging('mouse_move', 1, press))
    assert log == ['press', 'mouse_press', 'mouse_release', 'mouse_move']
    assert layer._mouse_drag_gen

    # the release of the button that started the drag ends it
    release = read_only_mouse_event(
        type='mouse_release', button=1, press_event=press
    )
    mouse_release_callbacks(layer, release)
    assert log[-1] == 'done'
    assert not layer._mouse_drag_gen


def test_new_drag_finishes_stale_drag():
    """A drag that lost its release finishes when the next drag starts."""
    layer = Points()
    log = []
    layer.mouse_drag_callbacks.append(_drag_logger(log))

    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    # the release went elsewhere, e.g. to another layer
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    assert log == ['press', 'release', 'press']
    assert len(layer._mouse_drag_gen) == 1


def test_failing_move_callback_does_not_stop_drag():
    layer = Points()
    log = []
    layer.mouse_drag_callbacks.append(_drag_logger(log))

    def broken_hover(layer, event):
        raise ValueError('boom on hover')

    layer.mouse_move_callbacks.append(broken_hover)
    mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    with pytest.warns(UserWarning, match='boom on hover'):
        mouse_move_callbacks(
            layer, read_only_mouse_event(type='mouse_move', is_dragging=True)
        )
    assert log == ['press', 'move']


def test_failing_generator_setup_warns():
    layer = Points()

    def drag(layer, event):
        raise ValueError('boom on press')
        yield

    layer.mouse_drag_callbacks.append(drag)
    with pytest.warns(UserWarning, match='boom on press'):
        mouse_press_callbacks(layer, read_only_mouse_event(type='mouse_press'))
    assert not layer._mouse_drag_gen
