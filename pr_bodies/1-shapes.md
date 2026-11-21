# References and relevant issues
Reported on Zulip: https://napari.zulipchat.com/#narrow/channel/212875-general/topic/Unselectable.20shape/near/637294286

# Description
Shapes vertices could be nearly impossible to grab, because the grab radius only used `layer.scale[-1]`. New Shapes layers copy the affine of the active layer, so this hit anyone with a scale stored in the affine (for example from a plugin reader).

Grab radius, nominally 5 px:

| Case | Before | After |
|---|---|---|
| Scale held in `affine` | 0 px | 5 px |
| Last data axis hidden by `dims.order`, scale `(1, 1, 100)` | 0 px | 5 px |
| Anisotropic scale `(0.004, 0.008)`, y / x | 2.5 / 5 px | 5 / 5 px |

To reproduce, make a square with `affine=np.diag([0.01, 0.01, 1])`, use Direct select, and try to drag a corner.

**Change to expected behavior:** nothing changes for an isotropic `scale`, no affine and the last axis displayed. With an anisotropic scale, the grab area is now larger along the smaller-scale axes. With an affine, the rotation handle length and highlight width also change, as they use the same conversion; these should now be right, but have no test.
