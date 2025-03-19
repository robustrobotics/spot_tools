# NOTE: This is from https://atsushisakai.github.io/PythonRobotics/index.html

"""
Path planning with Bezier curve.
author: Atsushi Sakai(@Atsushi_twi)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from numpy.typing import ArrayLike

show_animation = True


def average_yaw(yaw1_rad, yaw2_rad):
    # Ensure yaw values are within the range [0, 2*pi)
    yaw1_rad = yaw1_rad % (2 * math.pi)
    yaw2_rad = yaw2_rad % (2 * math.pi)
    # Calculate the midpoint of the arc between the two angles
    if abs(yaw2_rad - yaw1_rad) <= math.pi:
        avg_yaw_rad = (yaw1_rad + yaw2_rad) / 2
    else:
        avg_yaw_rad = (yaw1_rad + yaw2_rad + 2 * math.pi) / 2
    # Ensure the result is within the range [0, 2*pi)
    avg_yaw_rad = avg_yaw_rad % (2 * math.pi)
    return avg_yaw_rad


def get_heading(
    waypoints: ArrayLike,
    heading_mode: str = "leading",
):
    path = np.array(waypoints)
    yaw = np.array(
        [
            math.atan2(
                path[i + 1, 1] - path[i, 1],
                path[i + 1, 0] - path[i, 0],
            )
            for i in range(0, path.shape[0] - 1)
        ]
    )
    if (
        heading_mode == "leading"
    ):  # agent arrives at the waypoint lined up to the next one
        yaw = np.append(yaw, yaw[-1])  # yaw.append(yaw[-1])
    elif (
        heading_mode == "trailing"
    ):  # agent heading is aligned with the previous waypoint
        yaw = np.append(yaw[0], yaw)
    elif (
        heading_mode == "average"
    ):  # agent heading is aligned with the previous waypoint
        yawt = np.append(yaw, yaw[-1])  # yaw.append(yaw[-1])
        yawh = np.append(yaw[0], yaw)
        yawa = []
        for yaw1, yaw2 in zip(yawt, yawh):
            yawa.append(average_yaw(yaw1, yaw2))
        yaw = np.array(yawa)
    else:
        raise ValueError("Invalid heading mode: {}".format(heading_mode))

    path = np.concatenate((path[:, :2], yaw.reshape(-1, 1)), axis=1)
    return path


def smooth_path(
    path: ArrayLike, heading_mode="leading", n_points=4, offset=5.0
) -> ArrayLike:
    path = get_heading(path, heading_mode=heading_mode)

    smooth = []
    # loop through each pair of waypoints and compute a bazier spline between them
    for i in range(0, path.shape[0] - 1):
        sx, sy, syaw = path[i, :]
        ex, ey, eyaw = path[i + 1, :]
        intermediate_path, _ = calc_4points_bezier_path(
            sx, sy, syaw, ex, ey, eyaw, offset, n_points=n_points
        )
        intermediate_path = get_heading(intermediate_path, heading_mode="average")
        smooth.extend(intermediate_path)

    return np.array(smooth)


def calc_4points_bezier_path(sx, sy, syaw, ex, ey, eyaw, offset, n_points=100):
    """
    Compute control points and path given start and end position.
    :param sx: (float) x-coordinate of the starting point
    :param sy: (float) y-coordinate of the starting point
    :param syaw: (float) yaw angle at start
    :param ex: (float) x-coordinate of the ending point
    :param ey: (float) y-coordinate of the ending point
    :param eyaw: (float) yaw angle at the end
    :param offset: (float)
    :return: (numpy array, numpy array)
    """
    dist = np.hypot(sx - ex, sy - ey) / offset
    control_points = np.array(
        [
            [sx, sy],
            [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
            [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],
            [ex, ey],
        ]
    )

    path = calc_bezier_path(control_points, n_points=n_points)

    return path, control_points


def calc_bezier_path(control_points, n_points=100):
    """
    Compute bezier path (trajectory) given control points.
    :param control_points: (numpy array)
    :param n_points: (int) number of points in the trajectory
    :return: (numpy array)
    """
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def bernstein_poly(n, i, t):
    """
    Bernstein polynom.
    :param n: (int) polynom degree
    :param i: (int)
    :param t: (float)
    :return: (float)
    """
    return scipy.special.comb(n, i) * t**i * (1 - t) ** (n - i)


def bezier(t, control_points):
    """
    Return one point on the bezier curve.
    :param t: (float) number in [0, 1]
    :param control_points: (numpy array)
    :return: (numpy array) Coordinates of the point
    """
    n = len(control_points) - 1
    return np.sum(
        [bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0
    )


def bezier_derivatives_control_points(control_points, n_derivatives):
    """
    Compute control points of the successive derivatives of a given bezier curve.
    A derivative of a bezier curve is a bezier curve.
    See https://pomax.github.io/bezierinfo/#derivatives
    for detailed explanations
    :param control_points: (numpy array)
    :param n_derivatives: (int)
    e.g., n_derivatives=2 -> compute control points for first and second derivatives
    :return: ([numpy array])
    """
    w = {0: control_points}
    for i in range(n_derivatives):
        n = len(w[i])
        w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j]) for j in range(n - 1)])
    return w


def curvature(dx, dy, ddx, ddy):
    """
    Compute curvature at one point given first and second derivatives.
    :param dx: (float) First derivative along x axis
    :param dy: (float)
    :param ddx: (float) Second derivative along x axis
    :param ddy: (float)
    :return: (float)
    """
    return (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** (3 / 2)


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """Plot arrow."""
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(
            x,
            y,
            length * np.cos(yaw),
            length * np.sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
        )
        plt.plot(x, y)


def plot_curves(path, control_points):
    plt.plot(path.T[0], path.T[1])
    plt.plot(control_points.T[0], control_points.T[1])
    plt.show()


def main():
    """Plot an example bezier curve."""
    start_x = 10.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.radians(180.0)  # [rad]

    end_x = -0.0  # [m]
    end_y = -3.0  # [m]
    end_yaw = np.radians(-45.0)  # [rad]
    offset = 3.0

    path, control_points = calc_4points_bezier_path(
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset, n_points=4
    )

    path2 = calc_bezier_path(path, n_points=4)

    # Note: alternatively, instead of specifying start and end position
    # you can directly define n control points and compute the path:
    # control_points = np.array([[5., 1.], [-2.78, 1.], [-11.5, -4.5], [-6., -8.]])
    # path = calc_bezier_path(control_points, n_points=100)

    # Display the tangent, normal and radius of cruvature at a given point
    t = 0.86  # Number in [0, 1]
    x_target, y_target = bezier(t, control_points)
    derivatives_cp = bezier_derivatives_control_points(control_points, 2)
    point = bezier(t, control_points)
    dt = bezier(t, derivatives_cp[1])
    ddt = bezier(t, derivatives_cp[2])
    # Radius of curvature
    radius = 1 / curvature(dt[0], dt[1], ddt[0], ddt[1])
    # Normalize derivative
    dt /= np.linalg.norm(dt, 2)
    tangent = np.array([point, point + dt])
    normal = np.array([point, point + [-dt[1], dt[0]]])
    curvature_center = point + np.array([-dt[1], dt[0]]) * radius
    circle = plt.Circle(
        tuple(curvature_center), radius, color=(0, 0.8, 0.8), fill=False, linewidth=1
    )

    assert path.T[0][0] == start_x, "path is invalid"
    assert path.T[1][0] == start_y, "path is invalid"
    assert path.T[0][-1] == end_x, "path is invalid"
    assert path.T[1][-1] == end_y, "path is invalid"

    if show_animation:  # pragma: no cover
        fig, ax = plt.subplots()
        ax.plot(path.T[0], path.T[1], label="Bezier Path")
        ax.plot(path2.T[0], path2.T[1], label="Bezier Path 2")
        ax.plot(control_points.T[0], control_points.T[1], "--o", label="Control Points")
        ax.plot(x_target, y_target)
        ax.plot(tangent[:, 0], tangent[:, 1], label="Tangent")
        ax.plot(normal[:, 0], normal[:, 1], label="Normal")
        ax.add_artist(circle)
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        plt.show()


def main2():
    """Show the effect of the offset."""
    start_x = 10.0  # [m]
    start_y = 5.0  # [m]
    start_yaw = np.radians(180.0)  # [rad]

    end_x = -0.0  # [m]
    end_y = -0.0  # [m]
    end_yaw = np.radians(80.0)  # [rad]

    for offset in np.arange(1.0, 5.0, 1.0):
        path, control_points = calc_4points_bezier_path(
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset, n_points=4
        )
        assert path.T[0][0] == start_x, "path is invalid"
        assert path.T[1][0] == start_y, "path is invalid"
        assert path.T[0][-1] == end_x, "path is invalid"
        assert path.T[1][-1] == end_y, "path is invalid"

        if show_animation:  # pragma: no cover
            plt.plot(path.T[0], path.T[1], label="Offset=" + str(offset))

    if show_animation:  # pragma: no cover
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def main3():
    offset = 5.0
    waypoints_list = [
        [0.0, 0.0, 0.0],
        [0.6, 0.2, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.2, 1.0, 0.0],
        [1.0, 1.5, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ]
    path = smooth_path(
        waypoints_list, n_points=100, heading_mode="average", offset=offset
    )
    path2 = calc_bezier_path(np.array(waypoints_list)[:, :2], n_points=100)
    plt.plot(
        np.array(waypoints_list).T[0],
        np.array(waypoints_list).T[1],
        label="Offset=" + str(offset),
    )
    plt.plot(path.T[0], path.T[1], label="Offset=" + str(offset))
    plt.plot(path2.T[0], path2.T[1], label="Offset=" + str(offset))
    plt.show()


if __name__ == "__main__":
    main3()
