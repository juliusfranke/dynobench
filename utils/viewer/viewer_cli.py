from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

import argparse
import quad2d_viewer
import quad2dpole_viewer
import quad3d_viewer
import acrobot_viewer
import unicycle1_viewer
import unicycle2_viewer
import car_with_trailer_viewer
import integrator2_2d_viewer
import integrator1_2d_viewer
import robot_viewer
import sys


def get_robot_viewer(robot: str) -> robot_viewer.RobotViewer:
    viewer: robot_viewer.RobotViewer
    print("robot is")
    print(robot)
    if robot == "quad3d" or robot.startswith("quadrotor"):
        viewer = quad3d_viewer.Quad3dViewer()
    elif robot.startswith("unicycle1"):
        viewer = unicycle1_viewer.Unicycle1Viewer()
    elif robot.startswith("unicycle2"):
        viewer = unicycle2_viewer.Unicycle2Viewer()
    elif robot.startswith("quad2dpole"):
        viewer = quad2dpole_viewer.Quad2dpoleViewer()
    elif robot.startswith("quad2d"):
        viewer = quad2d_viewer.Quad2dViewer()
    elif robot.startswith("acrobot"):
        viewer = acrobot_viewer.AcrobotViewer()
    elif robot.startswith("car"):
        viewer = car_with_trailer_viewer.CarWithTrailerViewer()
    elif robot == "integrator2_2d":
        viewer = integrator2_2d_viewer.Integrator2_2dViewer()
    elif robot == "integrator1_2d":
        viewer = integrator1_2d_viewer.Integrator1_2dViewer()
    else:
        raise NotImplementedError("unknown model " + robot)
    return viewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", help="input file containing map")
    # parser.add_argument("--result", help="output file containing solution")
    # parser.add_argument("--result2", help="output file containing solution")
    parser.add_argument(
        "--robot", help="output file containing solution", required=True
    )
    args, unk = parser.parse_known_args()
    #
    viewer = get_robot_viewer(args.robot)

    # viewer_utils.check_viewer(
    # viewer, ["--env", args.env, "--result", args.result, "--result2",
    # args.result2])

    robot_viewer.check_viewer(viewer, unk)
