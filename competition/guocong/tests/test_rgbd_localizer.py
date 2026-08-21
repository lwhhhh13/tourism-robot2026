import numpy as np

from src.perception.rgbd_localizer import CameraIntrinsics, RGBDLocalizer, make_transform


def test_camera_info_and_mono16_mm_to_world():
    depth = np.full((5, 5), 2000, np.uint16)
    intrinsics = CameraIntrinsics.from_camera_info([100, 0, 2, 0, 100, 2, 0, 0, 1], 5, 5)
    transform = make_transform(np.eye(3), (1.0, 2.0, 3.0))
    camera, world, depth_m = RGBDLocalizer(patch_radius_px=1).localize_world(
        (2, 2), (0, 0, 4, 4), depth, intrinsics, transform
    )
    assert depth_m == 2.0
    np.testing.assert_allclose(camera, [0.0, 0.0, 2.0])
    np.testing.assert_allclose(world, [1.0, 2.0, 5.0])


def test_depth_median_ignores_zero_and_out_of_range():
    depth = np.array([[0, 1000, 0], [1000, 9000, 1000], [0, 1000, 0]], dtype=np.uint16)
    assert RGBDLocalizer(patch_radius_px=1).robust_depth_m(depth, (1, 1)) == 1.0
