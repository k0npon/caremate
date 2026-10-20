"""Occupancy-grid logic tests (synthetic depth, no rendering)."""
import numpy as np

from caremate.nbv_v0_7 import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, STATIONS, station_distance


def down_pose(x=1.4, y=-0.45, h=1.5):
    # MuJoCo camera looks along its -z; identity rotation => looking straight down.
    return {"pos": [x, y, h], "mat": np.eye(3).ravel().tolist(), "fovy_deg": 90.0}


def test_empty_table_becomes_free():
    g = OccupancyGrid()
    pose = down_pose()
    depth = np.full((120, 160), 1.5 - 0.76, np.float32)   # table surface everywhere
    g.integrate(depth, pose)
    c = g.counts()
    assert c["occupied"] == 0 and c["free"] > 0.5 * len(g.state)


def test_box_marks_occupied_and_hides_below():
    g = OccupancyGrid()
    pose = down_pose()
    depth = np.full((120, 160), 1.5 - 0.76, np.float32)
    depth[50:70, 70:90] = 1.5 - 0.95          # top of a 19 cm box under the camera
    g.integrate(depth, pose)
    assert g.counts()["occupied"] > 0
    # voxels directly under the box top stay unknown (occluded)
    below = (np.abs(g.centres[:, 0] - 1.4) < 0.02) & (np.abs(g.centres[:, 1] + 0.45) < 0.02) & \
            (g.centres[:, 2] < 0.9)
    assert (g.state[below] == UNKNOWN).all()


def test_gain_drops_after_observing():
    g = OccupancyGrid()
    pose = down_pose()
    before = g.expected_gain(pose)
    g.integrate(np.full((120, 160), 1.5 - 0.76, np.float32), pose)
    assert g.expected_gain(pose) < before


def test_station_distance_symmetric():
    assert station_distance("front", "front") == 0
    a, b = list(STATIONS)[:2]
    assert abs(station_distance(a, b) - station_distance(b, a)) < 1e-12
