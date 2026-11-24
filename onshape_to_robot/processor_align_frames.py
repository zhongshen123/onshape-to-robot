import fnmatch
import numpy as np
from .processor import Processor
from .config import Config
from .robot import Robot, Joint
from .message import info


class ProcessorAlignFrames(Processor):
    """
    Align frames processor.
    When enabled, every link frame is re-oriented to be aligned with the world
    (root) frame at the zero configuration (x forward, y left, z up). Joint axes
    are re-expressed accordingly, so the robot geometry and motion are unchanged.
    Frames (frame_*, closing_*) keep their own orientation.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # True, or a list of joint name patterns (child links of these joints are aligned)
        self.align_frames: bool | list = config.get("align_frames", False)

    def should_align(self, joint: Joint) -> bool:
        if self.align_frames == True:
            return True
        if isinstance(self.align_frames, list):
            return any(fnmatch.fnmatch(joint.name, entry) for entry in self.align_frames)
        return False

    def process(self, robot: Robot):
        if not self.align_frames:
            return

        count = 0
        for joint in robot.joints:
            if not self.should_align(joint):
                continue

            R_world_joint = joint.T_world_joint[:3, :3].copy()
            if np.allclose(R_world_joint, np.eye(3)):
                continue

            # Effective axis in the old joint frame (honours the "xyz" override)
            axis = joint.properties.get("xyz", joint.axis)
            if axis is None or axis is False:
                axis = (0, 0, 1)
            axis_world = R_world_joint @ np.asarray(axis, dtype=float)
            axis_world[np.abs(axis_world) < 1e-9] = 0.0

            # Joint (= child link) frame becomes world-aligned, same origin
            T_world_joint = joint.T_world_joint.copy()
            T_world_joint[:3, :3] = np.eye(3)
            joint.T_world_joint = T_world_joint
            joint.axis = axis_world

            # The "xyz" override was expressed in the old frame: it is now baked
            # into joint.axis. Copy so joints sharing a properties dict are unaffected.
            if "xyz" in joint.properties:
                joint.properties = {k: v for k, v in joint.properties.items() if k != "xyz"}
            count += 1

        print(info(f"* Aligned {count} joint/link frames with the world frame"))
