import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import animation


from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent))


from robot_viewer import RobotViewer
import viewer_utils

import sys

from pyplot3d2.uav import Uav
from pyplot3d2.utils import ypr_to_R

from scipy.spatial.transform import Rotation as RR


import numpy as np
from matplotlib.patches import FancyArrowPatch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


# For seamless integration we add the arrow3D method to the Axes3D class.


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
    return arrow


setattr(Axes3D, "arrow3D", _arrow3D)

# Example

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(0,2)

# ax.arrow3D(0,0,0,
#            1,1,1,
#            mutation_scale=20,
#            arrowstyle="-|>",
#            linestyle='dashed')

# ax.arrow3D(1,0,0,
#            1,1,1,
#            mutation_scale=20,
#            ec ='green',
#            fc='red')
# ax.set_title('3D Arrows Demo')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# fig.tight_layout()


class Robot:
    def __init__(self):
        pass

    def draw(self, ax, x, **kwargs):
        # ls = viewer_utils.plot_frame(ax, x, **kwargs)
        # self.lx = ls[0]
        # self.ly = ls[1]
        # self.lz = ls[2]
        (self.h,) = ax.plot(
            [x[0]], [x[1]], [x[2]], color=".5", linestyle="", marker=".", zorder=100
        )

        arm_length = 0.24  # in meters
        self.uav = Uav(ax, arm_length)
        q = x[3:7]
        p = np.array(x[:3])
        R_mat = RR.from_quat(q).as_matrix()
        self.uav.draw_at(p, R_mat, **kwargs, zorder=200)

        scale = 0.4
        self.arrow = ax.arrow3D(
            p[0],
            p[1],
            p[2],
            scale * R_mat[0, 2],
            scale * R_mat[1, 2],
            scale * R_mat[2, 2],
            mutation_scale=10,
            arrowstyle="-|>",
            # zorder=200,
            **kwargs,
        )
        # linestyle='dashed')

    #

    #

    def update(self, x, ax, uav, arrows):
        ll = viewer_utils.update_frame([self.lx, self.ly, self.lz], x)
        self.h.set_xdata([x[0]])
        self.h.set_ydata([x[1]])
        self.h.set_3d_properties([x[2]])
        q = x[3:7]
        p = np.array(x[:3])
        R_mat = RR.from_quat(q).as_matrix()

        arm_length = 0.24  # in meters
        _uav = Uav(ax, arm_length)

        uav[0].delete()
        uav[0] = _uav

        # print("uav at", p)
        _uav.draw_at(p, R_mat, color="blue")
        return ll + [self.h]

    def draw_traj_minimal(self, ax, Xs):
        xs = [p[0] for p in Xs]
        ys = [p[1] for p in Xs]

        # print("xs is")
        # print(xs)
        # ys =  []
        # for i,x in enumerate(xs):
        #     print(i,x)
        #     ys.append(x[1])
        #
        # ys = [p[1] for p in xs]
        zs = [p[2] for p in Xs]
        ax.plot3D(xs, ys, zs, "blue", alpha=0.5, zorder=100)
        # ax.plot3D(xs, ys, zs, 'blue')
        # ax.plot(xs, ys, zs,'bo')
        #         'blue')

    def draw_basic(self, ax, x):
        self.draw(ax, x)


class Quad3dViewer(RobotViewer):
    """ """

    def __init__(self):
        super().__init__(Robot)
        self.labels_x = [
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
        ]
        self.labels_u = ["f1", "f2", "f3", "f4"]

    def is_3d(self) -> bool:
        return True

    def view_problem(self, ax, env, **kwargs):
        if isinstance(env, str):
            with open(env, "r") as f:
                env = yaml.safe_load(f)

        print(env)
        lb = env["environment"]["min"]
        ub = env["environment"]["max"]
        obstacles = env["environment"]["obstacles"]
        start = np.array(env["robots"][0]["start"])
        goal = np.array(env["robots"][0]["goal"])

        for o in obstacles:
            if o["type"] == "box":
                viewer_utils.draw_cube(ax, o["center"], o["size"])
            if o["type"] == "sphere":
                viewer_utils.plt_sphere(ax, [o["center"]], [o["size"]])

        r = Robot()
        r.draw(ax, start, color="green")

        r = Robot()
        r.draw(ax, goal, color="red")

        ele = 30
        azm = -40

        if "recovery_with_obs" in env["name"]:
            ele = 10
            azm = -39

        if "window" in env["name"]:
            ele = 52
            azm = -31

        # For recovery with obstacles
        # azm = -39
        # ele = 10

        ax.view_init(elev=ele, azim=azm)  # Reproduce view
        ax.axes.set_xlim3d(left=lb[0], right=ub[0])
        ax.axes.set_ylim3d(bottom=lb[1], top=ub[1])
        ax.axes.set_zlim3d(bottom=lb[2], top=ub[2])

        # ax.set_box_aspect((1,1,1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # import matplotlib.pyplot as plt
        # import mpl_toolkits.mplot3d
        # import numpy as np

        # Functions from @Mateen Ulhaq and @karlo
        # TODO: maybe add this to the video!!!

        def set_axes_equal(ax: plt.Axes):
            """Set 3D plot axes to equal scale.

            Make axes of 3D plot have equal scale so that spheres appear as
            spheres and cubes as cubes.  Required since `ax.axis('equal')`
            and `ax.set_aspect('equal')` don't work on 3D.
            """
            limits = np.array(
                [
                    ax.get_xlim3d(),
                    ax.get_ylim3d(),
                    ax.get_zlim3d(),
                ]
            )
            origin = np.mean(limits, axis=1)
            radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
            _set_axes_radius(ax, origin, radius)

        def _set_axes_radius(ax, origin, radius):
            x, y, z = origin
            ax.set_xlim3d([x - radius, x + radius])
            ax.set_ylim3d([y - radius, y + radius])
            ax.set_zlim3d([z - radius, z + radius])

        # # Generate and plot a unit sphere
        # u = np.linspace(0, 2*np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # x = np.outer(np.cos(u), np.sin(v)) # np.outer() -> outer vector product
        # y = np.outer(np.sin(u), np.sin(v))
        # z = np.outer(np.ones(np.size(u)), np.cos(v))
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.plot_surface(x, y, z)

        ax.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line
        # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown
        # in image above)
        set_axes_equal(ax)  # IMPORTANT - this is also required

    # def view_trajectory(self,ax,result, **kwargs):
    #     """
    #     """
    #
    #     if isinstance(result, str):
    #         with open(result) as f:
    #             __result = yaml.safe_load(f)
    #             result = __result["result"][0]
    #
    #     states = result["states"]
    #     xs = [p[0] for p in states]
    #     ys = [p[1] for p in states]
    #     zs = [p[2] for p in states]
    #     ax.plot3D(xs, ys, zs, 'gray')
    #
    #     plot_orientation_every = 50
    #     for i in range(0, len(states) , plot_orientation_every):
    #         print(f"states[i] {states[i]}")
    #         r = Robot()
    #         r.draw(ax,states[i])

    # def view_state(self,ax,state, facecolor='none' , edgecolor='black', **kwargs):
    #     """
    #     """
    #     r = Robot()
    #     r.draw(ax,state)

    # def plot_traj(self,axs,result, **kwargs):
    #     """
    #     """
    #
    #     xs = result["states"]
    #     us = result["actions"]
    #
    #     for i,l in enumerate( labels_x) :
    #         xi = [ x[i] for x in xs]
    #         axs[0].plot(xi, label = l)
    #     axs[0].legend()
    #
    #     for i,l in enumerate( labels_u) :
    #         ui = [ u[i] for u in us]
    #         axs[1].plot(ui, label = l)
    #     axs[1].legend()

    def view_trajectory(self, ax, result, **kwargs):
        viewer_utils.draw_traj_default(
            ax, result, self.RobotDrawerClass, draw_basic_every=1000
        )

    def make_video(
        self, env, result, filename_video: str = "", interactive: bool = False
    ):
        # fig = plt.figure(figsize=(6,6)) # points appear too big!
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")
        self.view_problem(ax, env)
        # ax.set_title(env["name"])
        if isinstance(result, str):
            with open(result) as f:
                __result = yaml.safe_load(f)
                result = __result["result"][0]

        states = result["states"]

        r = Robot()
        r.draw(ax, states[0], color="blue")

        states = result["states"]
        print("hello")
        print(f"states {states}")

        r.draw_traj_minimal(ax, states)

        ax.axis("off")
        plt.tight_layout()

        def animate_func(i):
            """ """
            state = states[i]
            r.uav.delete()
            r.h.remove()
            r.arrow.remove()
            r.draw(ax, state, color="blue")

            # r.
            # return r.update(state, ax, uavs, arrows)

        T = len(states)

        anim = animation.FuncAnimation(
            fig, animate_func, frames=T, interval=10, blit=False
        )

        # plt.show()
        DPI = 200
        if len(filename_video):
            speed = 1
            print(f"saving video: {filename_video}")
            # anim.save(filename_video, "ffmpeg", fps=10 * speed, dpi=100)
            anim.save(filename_video, "ffmpeg", fps=10 * speed, dpi=DPI)
            print(f"saving video: {filename_video} -- DONE")

        elif interactive:
            plt.show()


if __name__ == "__main__":
    viewer = Quad3dViewer()
    viewer_utils.check_viewer(viewer, is_3d=True)
