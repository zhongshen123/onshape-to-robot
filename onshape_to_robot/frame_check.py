"""
Frame convention checker (and fixer) for exported robot files.

Convention checked, at the zero configuration (all joints at 0):

* every link frame has the same orientation as the root/world frame,
  i.e. x forward, y left, z up;
* every joint axis is along +-X, +-Y or +-Z of that frame.

Supported formats: URDF (.urdf) and MuJoCo MJCF (.xml).

Command line: onshape-to-robot-check <robot.urdf | robot.xml | robot_dir> [--fix]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import quat2mat, mat2quat, axangle2quat

from .message import bright, error, info, success, warning

AXIS_NAMES = ["X", "Y", "Z"]
DEFAULT_TOLERANCE_DEG = 0.5

# A link with less mass than this, no geometry and no children is a "frame"
# (frame_xxx / closing_xxx dummy link in URDF)
FRAME_MASS_EPS = 1e-6


# ----------------------------------------------------------------------------
# Small math helpers
# ----------------------------------------------------------------------------


def make_T(R: np.ndarray | None = None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle (deg) of the rotation R, i.e. how far it is from identity."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def principal_axis(v, tolerance_deg: float) -> tuple[str | None, float]:
    """
    Returns ("+X", angle) when v is within tolerance of a principal axis,
    otherwise (None, angle to the closest principal axis).
    """
    v = normalize(v)
    if np.linalg.norm(v) < 1e-12:
        return None, 180.0
    k = int(np.argmax(np.abs(v)))
    angle = float(np.degrees(np.arccos(np.clip(abs(v[k]), -1.0, 1.0))))
    label = ("+" if v[k] > 0 else "-") + AXIS_NAMES[k]
    return (label if angle <= tolerance_deg else None), angle


def format_vector(v) -> str:
    v = normalize(v)
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def describe_axis(v, tolerance_deg: float) -> str:
    label, angle = principal_axis(v, tolerance_deg)
    if label:
        return label
    return f"{format_vector(v)} ({angle:.1f} deg off nearest axis)"


def describe_frame(R: np.ndarray, tolerance_deg: float) -> str:
    """e.g. 'x->+Z y->-X z->-Y': where each local axis points in the world."""
    return "  ".join(
        f"{name.lower()}->{describe_axis(R[:, k], tolerance_deg)}"
        for k, name in enumerate(AXIS_NAMES)
    )


def fmt(x: float) -> str:
    if abs(x) < 1e-12:
        x = 0.0
    s = "%.10g" % x
    return "0" if s == "-0" else s


def fmt_vec(values) -> str:
    return " ".join(fmt(float(x)) for x in values)


def snap(v, eps: float = 1e-9) -> np.ndarray:
    v = np.array(v, dtype=float)
    v[np.abs(v) < eps] = 0.0
    return v


def parse_vec(text: str | None, default) -> np.ndarray:
    if text is None or text.strip() == "":
        return np.array(default, dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def make_parser() -> ET.XMLParser:
    return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))


def write_tree(tree: ET.ElementTree, source: str, filename: str):
    """
    Writes tree, keeping the source file's prolog (XML declaration and the
    comments before the root element) and its self-closing tag style.
    """
    with open(source, "r", encoding="utf-8") as stream:
        original = stream.read()
    root_tag = tree.getroot().tag
    start = original.find("<" + root_tag)
    prolog = original[:start] if start >= 0 else ""
    body = ET.tostring(tree.getroot(), encoding="unicode")
    if original.count(" />") < original.count("/>") / 2:
        body = body.replace(" />", "/>")
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(prolog + body + "\n")


# ----------------------------------------------------------------------------
# Format independent model used for checking
# ----------------------------------------------------------------------------


@dataclass
class FrameInfo:
    name: str
    kind: str  # "link", "body", "frame" (URDF dummy link), "site"
    R_world: np.ndarray


@dataclass
class JointInfo:
    name: str
    joint_type: str
    child: str
    axis_world: np.ndarray | None  # None for fixed / free / ball joints


@dataclass
class Model:
    filename: str
    format: str  # "urdf" or "mujoco"
    root: str
    frames: list[FrameInfo] = field(default_factory=list)
    joints: list[JointInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Issue:
    kind: str  # "frame" or "axis"
    name: str
    message: str


@dataclass
class Report:
    model: Model
    issues: list[Issue]
    checked_frames: int
    checked_joints: int
    tolerance_deg: float

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


# ----------------------------------------------------------------------------
# URDF
# ----------------------------------------------------------------------------


def urdf_origin(element: ET.Element | None) -> np.ndarray:
    """Returns the 4x4 transform of an <origin> child of element (identity if absent)."""
    if element is None:
        return np.eye(4)
    origin = element.find("origin")
    if origin is None:
        return np.eye(4)
    xyz = parse_vec(origin.get("xyz"), [0, 0, 0])
    rpy = parse_vec(origin.get("rpy"), [0, 0, 0])
    return make_T(euler2mat(*rpy, axes="sxyz"), xyz)


def urdf_set_origin(element: ET.Element, T: np.ndarray):
    """Writes T into the <origin> child of element (created if needed)."""
    origin = element.find("origin")
    if origin is None:
        origin = ET.Element("origin")
        origin.tail = element.text
        element.insert(0, origin)
    rpy = snap(mat2euler(T[:3, :3], axes="sxyz"), 1e-12)
    origin.set("xyz", fmt_vec(snap(T[:3, 3], 1e-12)))
    origin.set("rpy", fmt_vec(rpy))


class URDFRobot:
    def __init__(self, filename: str):
        self.filename = filename
        self.tree = ET.parse(filename, parser=make_parser())
        self.robot = self.tree.getroot()
        if self.robot.tag != "robot":
            raise ValueError(f"{filename}: root element is <{self.robot.tag}>, expected <robot>")

        self.links: dict[str, ET.Element] = {
            link.get("name"): link for link in self.robot.findall("link")
        }
        self.joints: list[ET.Element] = self.robot.findall("joint")
        self.parent_joint: dict[str, ET.Element] = {}
        self.child_joints: dict[str, list[ET.Element]] = {name: [] for name in self.links}
        for joint in self.joints:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            self.parent_joint[child] = joint
            self.child_joints.setdefault(parent, []).append(joint)

        roots = [name for name in self.links if name not in self.parent_joint]
        if len(roots) != 1:
            raise ValueError(f"{filename}: expected exactly one root link, found {roots}")
        self.root = roots[0]

        # World (root) transform of every link at the zero configuration
        self.T_world: dict[str, np.ndarray] = {self.root: np.eye(4)}
        self.order: list[str] = [self.root]  # parents before children
        stack = [self.root]
        while stack:
            name = stack.pop()
            for joint in self.child_joints.get(name, []):
                child = joint.find("child").get("link")
                self.T_world[child] = self.T_world[name] @ urdf_origin(joint)
                self.order.append(child)
                stack.append(child)

    def joint_axis(self, joint: ET.Element) -> np.ndarray:
        axis = joint.find("axis")
        return parse_vec(axis.get("xyz") if axis is not None else None, [1, 0, 0])

    def is_frame(self, name: str) -> bool:
        """Dummy link used for a frame (frame_xxx, closing_xxx): no geometry, no mass, leaf."""
        link = self.links[name]
        joint = self.parent_joint.get(name)
        if joint is None or joint.get("type") != "fixed":
            return False
        if link.find("visual") is not None or link.find("collision") is not None:
            return False
        if self.child_joints.get(name):
            return False
        inertial = link.find("inertial")
        if inertial is not None and inertial.find("mass") is not None:
            if float(inertial.find("mass").get("value", "0")) > FRAME_MASS_EPS:
                return False
        return True

    def model(self) -> Model:
        model = Model(self.filename, "urdf", self.root)
        for name in self.order:
            kind = "frame" if self.is_frame(name) else "link"
            model.frames.append(FrameInfo(name, kind, self.T_world[name][:3, :3]))
        for joint in self.joints:
            child = joint.find("child").get("link")
            joint_type = joint.get("type")
            axis_world = None
            if joint_type in ("revolute", "continuous", "prismatic", "planar"):
                axis_world = self.T_world[child][:3, :3] @ self.joint_axis(joint)
            model.joints.append(JointInfo(joint.get("name"), joint_type, child, axis_world))
        return model

    def align(self, include_frames: bool = False, ignore: list[str] = [], tolerance_deg: float = DEFAULT_TOLERANCE_DEG):
        """
        Re-orients link frames that are more than tolerance_deg off so they are
        world aligned, without changing where anything is in the world
        (geometry, inertia, joint axes, frames).
        """
        new_T: dict[str, np.ndarray] = {}
        for name in self.order:
            T = self.T_world[name]
            keep = (
                (self.is_frame(name) and not include_frames)
                or matches(name, ignore)
                or rotation_angle_deg(T[:3, :3]) <= tolerance_deg
            )
            new_T[name] = T.copy() if keep else make_T(np.eye(3), T[:3, 3])

        for joint in self.joints:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            T_new = np.linalg.inv(new_T[parent]) @ new_T[child]
            if not np.allclose(T_new, urdf_origin(joint), atol=1e-12):
                urdf_set_origin(joint, T_new)
            # Joint axis is expressed in the child frame
            R_new_old = new_T[child][:3, :3].T @ self.T_world[child][:3, :3]
            axis = joint.find("axis")
            if np.allclose(R_new_old, np.eye(3)):
                pass
            elif axis is not None and joint.get("type") != "fixed":
                axis.set("xyz", fmt_vec(snap(R_new_old @ self.joint_axis(joint))))
            elif axis is None and joint.get("type") in ("revolute", "continuous", "prismatic", "planar"):
                axis = ET.SubElement(joint, "axis")
                axis.set("xyz", fmt_vec(snap(R_new_old @ np.array([1.0, 0, 0]))))

        for name, link in self.links.items():
            R_new_old = new_T[name][:3, :3].T @ self.T_world[name][:3, :3]
            if np.allclose(R_new_old, np.eye(3)):
                continue
            for element in list(link.findall("visual")) + list(link.findall("collision")) + list(
                link.findall("inertial")
            ):
                T = urdf_origin(element)
                urdf_set_origin(element, make_T(R_new_old, np.zeros(3)) @ T)


    def write(self, filename: str):
        write_tree(self.tree, self.filename, filename)


# ----------------------------------------------------------------------------
# MuJoCo (MJCF)
# ----------------------------------------------------------------------------

ORIENTATION_ATTRS = ("quat", "axisangle", "euler", "xyaxes", "zaxis")
POSED_CHILDREN = ("geom", "site", "camera", "light", "inertial")


class MJCFRobot:
    def __init__(self, filename: str):
        self.filename = filename
        self.tree = ET.parse(filename, parser=make_parser())
        self.mujoco = self.tree.getroot()
        if self.mujoco.tag != "mujoco":
            raise ValueError(f"{filename}: root element is <{self.mujoco.tag}>, expected <mujoco>")
        self.has_includes = self._resolve_includes(self.mujoco, os.path.dirname(os.path.abspath(filename)))

        compiler = self.mujoco.find("compiler")
        self.degrees = compiler is None or compiler.get("angle", "degree") != "radian"
        self.eulerseq = compiler.get("eulerseq", "xyz") if compiler is not None else "xyz"

        self._read_defaults()

        self.worldbody = self.mujoco.find("worldbody")
        if self.worldbody is None:
            raise ValueError(f"{filename}: no <worldbody>")
        self.has_frame_elements = self.worldbody.find(".//frame") is not None

        # World transform of every body / element at qpos = 0
        self.T_world: dict[ET.Element, np.ndarray] = {}
        self.parent_of: dict[ET.Element, ET.Element] = {}
        self.bodies: list[ET.Element] = []
        self._walk(self.worldbody, np.eye(4), "main")

    # --- includes / defaults ------------------------------------------------

    def _resolve_includes(self, element: ET.Element, directory: str) -> bool:
        found = False
        for k, child in enumerate(list(element)):
            if child.tag == "include":
                found = True
                path = os.path.join(directory, child.get("file"))
                included = ET.parse(path, parser=make_parser()).getroot()
                self._resolve_includes(included, os.path.dirname(path))
                element.remove(child)
                for j, sub in enumerate(list(included)):
                    # Merge into an existing section of the same name if any (e.g. worldbody)
                    existing = element.find(sub.tag) if sub.tag in ("worldbody",) else None
                    if existing is not None:
                        existing.extend(list(sub))
                    else:
                        element.insert(k + j, sub)
            elif isinstance(child.tag, str):
                found = self._resolve_includes(child, directory) or found
        return found

    def _read_defaults(self):
        """class name -> tag -> attributes (with inheritance)"""
        self.defaults: dict[str, dict[str, dict[str, str]]] = {}

        def visit(default: ET.Element, inherited: dict, name: str):
            current = {tag: dict(attrs) for tag, attrs in inherited.items()}
            for child in default:
                if isinstance(child.tag, str) and child.tag != "default":
                    current.setdefault(child.tag, {}).update(child.attrib)
            self.defaults[name] = current
            for child in default.findall("default"):
                visit(child, current, child.get("class"))

        for default in self.mujoco.findall("default"):
            visit(default, {}, default.get("class", "main"))

    def attr(self, element: ET.Element, key: str, cls: str):
        if key in element.attrib:
            return element.get(key)
        cls = element.get("class", cls)
        return self.defaults.get(cls, {}).get(element.tag, {}).get(key)

    # --- poses --------------------------------------------------------------

    def orientation(self, element: ET.Element) -> np.ndarray:
        if element.get("quat") is not None:
            return quat2mat(normalize(parse_vec(element.get("quat"), [1, 0, 0, 0])))
        if element.get("axisangle") is not None:
            v = parse_vec(element.get("axisangle"), [0, 0, 1, 0])
            angle = np.radians(v[3]) if self.degrees else v[3]
            return quat2mat(axangle2quat(v[:3], angle))
        if element.get("euler") is not None:
            angles = parse_vec(element.get("euler"), [0, 0, 0])
            if self.degrees:
                angles = np.radians(angles)
            return _euler_mujoco(angles, self.eulerseq)
        if element.get("xyaxes") is not None:
            v = parse_vec(element.get("xyaxes"), [1, 0, 0, 0, 1, 0])
            x = normalize(v[:3])
            y = normalize(v[3:] - np.dot(v[3:], x) * x)
            return np.column_stack([x, y, np.cross(x, y)])
        if element.get("zaxis") is not None:
            z = normalize(parse_vec(element.get("zaxis"), [0, 0, 1]))
            return quat2mat(_quat_from_z(z))
        return np.eye(3)

    def local_T(self, element: ET.Element) -> np.ndarray:
        return make_T(self.orientation(element), parse_vec(element.get("pos"), [0, 0, 0]))

    def _walk(self, element: ET.Element, T_world_element: np.ndarray, cls: str):
        for child in element:
            if not isinstance(child.tag, str):
                continue
            if child.tag in ("body", "frame"):
                T = T_world_element @ self.local_T(child)
                self.T_world[child] = T
                self.parent_of[child] = element
                if child.tag == "body":
                    self.bodies.append(child)
                self._walk(child, T, child.get("childclass", cls))
            elif child.tag in POSED_CHILDREN or child.tag == "joint":
                self.parent_of[child] = element
                self.T_world[child] = T_world_element @ self.local_T(child)

    def body_class(self, element: ET.Element) -> str:
        cls = "main"
        chain = []
        node = element
        while node in self.parent_of:
            chain.append(node)
            node = self.parent_of[node]
        for node in reversed(chain):
            if node.tag in ("body", "frame") and node.get("childclass"):
                cls = node.get("childclass")
        return cls

    def joint_axis_world(self, joint: ET.Element) -> np.ndarray:
        cls = self.body_class(self.parent_of[joint])
        axis = parse_vec(self.attr(joint, "axis", cls), [0, 0, 1])
        return self.T_world[self.parent_of[joint]][:3, :3] @ axis

    def joint_type(self, joint: ET.Element) -> str:
        cls = self.body_class(self.parent_of[joint])
        return self.attr(joint, "type", cls) or "hinge"

    def model(self) -> Model:
        root_bodies = [b for b in self.worldbody.findall("body")]
        model = Model(self.filename, "mujoco", ", ".join(b.get("name", "?") for b in root_bodies))
        for body in self.bodies:
            model.frames.append(FrameInfo(body.get("name", "<unnamed body>"), "body", self.T_world[body][:3, :3]))
        for element, T in self.T_world.items():
            if element.tag == "site":
                model.frames.append(FrameInfo(element.get("name", "<unnamed site>"), "site", T[:3, :3]))
        for element in self.T_world:
            if element.tag == "joint":
                joint_type = self.joint_type(element)
                axis = self.joint_axis_world(element) if joint_type in ("hinge", "slide") else None
                body = self.parent_of[element]
                model.joints.append(JointInfo(element.get("name", "<unnamed joint>"), joint_type, body.get("name", "?"), axis))
            elif element.tag == "freejoint":
                pass
        if self.has_frame_elements:
            model.notes.append("file uses <frame> elements: they are taken into account for checking")
        return model

    # --- fixing ---------------------------------------------------------------

    def align(self, include_frames: bool = False, ignore: list[str] = [], tolerance_deg: float = DEFAULT_TOLERANCE_DEG):
        if self.has_includes:
            raise ValueError("file uses <include>: run the fix on the included robot file itself (e.g. robot.xml)")
        if self.has_frame_elements:
            raise ValueError("file uses <frame> elements, which --fix does not support")
        for eq in self.mujoco.findall("equality/weld"):
            if eq.get("body1") is not None and eq.get("relpose") is not None:
                raise ValueError("<weld> with body1/relpose is not supported by --fix")

        new_R: dict[ET.Element, np.ndarray] = {}
        for body in self.bodies:
            keep = matches(body.get("name", ""), ignore) or (
                rotation_angle_deg(self.T_world[body][:3, :3]) <= tolerance_deg
            )
            new_R[body] = self.T_world[body][:3, :3].copy() if keep else np.eye(3)

        # Re-express everything that lives in a body frame
        for body in self.bodies:
            R_old = self.T_world[body][:3, :3]
            R_new = new_R[body]
            R_new_old = R_new.T @ R_old  # old body frame -> new body frame
            parent = self.parent_of[body]
            T_parent_new = (
                make_T(new_R[parent], self.T_world[parent][:3, 3]) if parent in new_R else np.eye(4)
            )
            T_body_new = make_T(R_new, self.T_world[body][:3, 3])
            T_local_new = np.linalg.inv(T_parent_new) @ T_body_new
            if not np.allclose(T_local_new, self.local_T(body), atol=1e-12):
                set_pose(body, T_local_new)

            if np.allclose(R_new_old, np.eye(3)):
                # Body kept: sites may still need aligning (--include-frames)
                if include_frames:
                    for child in body.findall("site"):
                        if rotation_angle_deg(R_old @ self.orientation(child)) > tolerance_deg:
                            set_orientation(child, R_old.T)
                continue

            cls = self.body_class(body)
            for child in body:
                if not isinstance(child.tag, str):
                    continue
                if child.tag == "joint":
                    if child.get("pos") is not None:
                        child.set("pos", fmt_vec(snap(R_new_old @ parse_vec(child.get("pos"), [0, 0, 0]))))
                    if self.joint_type(child) in ("hinge", "slide"):
                        axis = parse_vec(self.attr(child, "axis", cls), [0, 0, 1])
                        child.set("axis", fmt_vec(snap(R_new_old @ axis)))
                elif child.tag in POSED_CHILDREN:
                    self._rotate_element(child, R_new_old, R_old, include_frames, tolerance_deg)

        # Equality anchors expressed in body1 frame
        name_to_body = {b.get("name"): b for b in self.bodies}
        for tag in ("connect", "weld"):
            for eq in self.mujoco.findall(f"equality/{tag}"):
                body1 = name_to_body.get(eq.get("body1"))
                if body1 is not None and eq.get("anchor") is not None:
                    R_new_old = new_R[body1].T @ self.T_world[body1][:3, :3]
                    eq.set("anchor", fmt_vec(snap(R_new_old @ parse_vec(eq.get("anchor"), [0, 0, 0]))))

    def _rotate_element(
        self,
        element: ET.Element,
        R_new_old: np.ndarray,
        R_body_world_old: np.ndarray,
        include_frames: bool,
        tolerance_deg: float,
    ):
        if element.get("pos") is not None:
            element.set("pos", fmt_vec(snap(R_new_old @ parse_vec(element.get("pos"), [0, 0, 0]), 1e-12)))
        if element.get("fromto") is not None:
            # fromto fully defines position and orientation
            v = parse_vec(element.get("fromto"), [0] * 6)
            element.set("fromto", fmt_vec(snap(np.concatenate([R_new_old @ v[:3], R_new_old @ v[3:]]), 1e-12)))
            return
        if element.tag == "light" and element.get("dir") is not None:
            element.set("dir", fmt_vec(snap(R_new_old @ parse_vec(element.get("dir"), [0, 0, -1]))))
        if element.tag == "inertial" and element.get("fullinertia") is not None:
            a = parse_vec(element.get("fullinertia"), [0] * 6)
            inertia = np.array([[a[0], a[3], a[4]], [a[3], a[1], a[5]], [a[4], a[5], a[2]]])
            inertia = R_new_old @ inertia @ R_new_old.T
            element.set(
                "fullinertia",
                fmt_vec(
                    snap(
                        [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]],
                        1e-15,
                    )
                ),
            )
            return
        if element.tag == "site" and include_frames and rotation_angle_deg(
            R_body_world_old @ self.orientation(element)
        ) > tolerance_deg:
            R_element_new = np.eye(3)
        else:
            R_element_new = R_new_old @ self.orientation(element)
        has_orientation = any(element.get(a) is not None for a in ORIENTATION_ATTRS)
        if has_orientation or not np.allclose(R_element_new, np.eye(3)):
            set_orientation(element, R_element_new)

    def write(self, filename: str):
        write_tree(self.tree, self.filename, filename)


def _euler_mujoco(angles, seq: str) -> np.ndarray:
    """MuJoCo euler convention: lowercase = intrinsic (moving) axes, uppercase = extrinsic."""
    R = np.eye(3)
    for angle, axis in zip(angles, seq):
        k = "xyz".index(axis.lower())
        v = np.zeros(3)
        v[k] = 1.0
        Rk = quat2mat(axangle2quat(v, angle))
        R = R @ Rk if axis.islower() else Rk @ R
    return R


def _quat_from_z(z: np.ndarray) -> np.ndarray:
    """Minimal rotation taking (0,0,1) to z (MuJoCo zaxis semantics)."""
    z0 = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z0, z))
    if c > 1 - 1e-12:
        return np.array([1.0, 0, 0, 0])
    if c < -1 + 1e-12:
        return np.array([0.0, 1, 0, 0])
    axis = np.cross(z0, z)
    return axangle2quat(axis, np.arccos(c))


def set_orientation(element: ET.Element, R: np.ndarray):
    for a in ORIENTATION_ATTRS:
        if a in element.attrib:
            del element.attrib[a]
    if not np.allclose(R, np.eye(3), atol=1e-12):
        q = mat2quat(R)
        if q[0] < 0:
            q = -q
        element.set("quat", fmt_vec(snap(q, 1e-12)))


def set_pose(element: ET.Element, T: np.ndarray):
    element.set("pos", fmt_vec(snap(T[:3, 3], 1e-12)))
    set_orientation(element, T[:3, :3])


# ----------------------------------------------------------------------------
# Checking
# ----------------------------------------------------------------------------


def matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def load(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".urdf":
        return URDFRobot(filename)
    if ext == ".xml":
        return MJCFRobot(filename)
    raise ValueError(f"{filename}: unsupported extension (expected .urdf or .xml)")


def check_model(
    model: Model,
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
    include_frames: bool = False,
    ignore: list[str] = [],
) -> Report:
    issues: list[Issue] = []
    checked_frames = 0
    for frame in model.frames:
        if frame.kind in ("frame", "site") and not include_frames:
            continue
        if matches(frame.name, ignore):
            continue
        checked_frames += 1
        angle = rotation_angle_deg(frame.R_world)
        if angle > tolerance_deg:
            issues.append(
                Issue(
                    "frame",
                    frame.name,
                    f"{frame.kind} frame rotated {angle:.1f} deg from world: {describe_frame(frame.R_world, tolerance_deg)}",
                )
            )

    checked_joints = 0
    for joint in model.joints:
        if joint.axis_world is None or matches(joint.name, ignore):
            continue
        checked_joints += 1
        label, angle = principal_axis(joint.axis_world, tolerance_deg)
        if label is None:
            issues.append(
                Issue(
                    "axis",
                    joint.name,
                    f"axis in world is {format_vector(joint.axis_world)}, {angle:.1f} deg off the nearest of +-X/Y/Z"
                    " (re-orienting frames cannot fix this: check the mate in Onshape)",
                )
            )

    return Report(model, issues, checked_frames, checked_joints, tolerance_deg)


def check_file(filename: str, tolerance_deg=DEFAULT_TOLERANCE_DEG, include_frames=False, ignore=[]) -> Report:
    return check_model(load(filename).model(), tolerance_deg, include_frames, ignore)


def print_report(report: Report, verbose: bool = False):
    model = report.model
    print(bright(f"* Frame check: {model.filename} ({model.format}, root: {model.root})"))
    print(
        info(
            f"  Convention: all frames world-aligned at zero pose (x forward, y left, z up), "
            f"joint axes along +-X/Y/Z, tolerance {report.tolerance_deg:g} deg"
        )
    )
    for note in model.notes:
        print(info(f"  Note: {note}"))

    if verbose:
        for joint in model.joints:
            if joint.axis_world is not None:
                print(f"    joint {joint.name:30s} axis {describe_axis(joint.axis_world, report.tolerance_deg)}")

    frame_issues = [i for i in report.issues if i.kind == "frame"]
    axis_issues = [i for i in report.issues if i.kind == "axis"]
    for issue in frame_issues:
        print(warning(f"  [frame] {issue.name}: {issue.message}"))
    for issue in axis_issues:
        print(error(f"  [axis]  {issue.name}: {issue.message}"))

    summary = (
        f"{report.checked_frames} frames, {report.checked_joints} joint axes checked: "
        f"{len(frame_issues)} misaligned frames, {len(axis_issues)} tilted axes"
    )
    if report.ok:
        print(success(f"  OK - {summary}"))
    else:
        print(warning(f"  FAILED - {summary}"))
        if frame_issues and not axis_issues:
            print(info("  Misaligned frames can be fixed automatically: use --fix, or \"align_frames\": true in config.json"))


def fix_file(
    filename: str,
    output: str,
    include_frames: bool = False,
    ignore: list[str] = [],
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
):
    """Writes a copy of filename where every link frame more than tolerance_deg off is world-aligned."""
    robot = load(filename)
    robot.align(include_frames=include_frames, ignore=ignore, tolerance_deg=tolerance_deg)
    robot.write(output)


def find_robot_file(path: str) -> str:
    """Accepts a file, or a robot directory containing config.json."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise ValueError(f"{path} does not exist")
    config_file = os.path.join(path, "config.json")
    if os.path.exists(config_file):
        import commentjson

        with open(config_file, "r", encoding="utf8") as stream:
            config = commentjson.load(stream)
        output_format = config.get("output_format", config.get("outputFormat"))
        filename = config.get("output_filename", config.get("outputFilename", "robot"))
        ext = {"urdf": "urdf", "mujoco": "xml"}.get(output_format)
        if ext is None:
            raise ValueError(f"output_format '{output_format}' is not supported by the frame checker (urdf or mujoco)")
        candidate = os.path.join(path, f"{filename}.{ext}")
        if os.path.exists(candidate):
            return candidate
        raise ValueError(f"{candidate} not found: run onshape-to-robot first")
    for candidate in ("robot.urdf", "robot.xml"):
        if os.path.exists(os.path.join(path, candidate)):
            return os.path.join(path, candidate)
    raise ValueError(f"no robot.urdf / robot.xml / config.json found in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check that every link frame is world-aligned (x forward, y left, z up) at zero pose "
        "and every joint axis is along +-X/Y/Z. Works on URDF and MuJoCo files."
    )
    parser.add_argument("path", help="robot.urdf, robot.xml, or a robot directory with config.json")
    parser.add_argument("--fix", action="store_true", help="write a corrected copy with world-aligned frames")
    parser.add_argument("-o", "--output", help="output file for --fix (default: <name>_aligned.<ext>)")
    parser.add_argument("--in-place", action="store_true", help="with --fix, overwrite the input file")
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE_DEG, help="angle tolerance in degrees (default 0.5)"
    )
    parser.add_argument(
        "--include-frames",
        action="store_true",
        help="also check (and with --fix, align) frame_*/closing_* frames and MuJoCo sites",
    )
    parser.add_argument(
        "--ignore", nargs="*", default=[], help="link/joint name patterns to skip, e.g. --ignore 'camera*'"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="also list every joint axis")
    args = parser.parse_args()

    try:
        filename = find_robot_file(args.path)
        report = check_file(filename, args.tolerance, args.include_frames, args.ignore)
        print_report(report, args.verbose)

        if not args.fix:
            sys.exit(0 if report.ok else 1)

        if args.in_place:
            output = filename
        elif args.output:
            output = args.output
        else:
            stem, ext = os.path.splitext(filename)
            output = f"{stem}_aligned{ext}"
        if os.path.abspath(output) != os.path.abspath(filename) and os.path.dirname(
            os.path.abspath(output)
        ) != os.path.dirname(os.path.abspath(filename)):
            print(warning("  Warning: output is in another directory, relative mesh paths may break"))

        fix_file(filename, output, args.include_frames, args.ignore, args.tolerance)
        print(success(f"* Wrote {output}"))
        fixed = check_file(output, args.tolerance, args.include_frames, args.ignore)
        print_report(fixed, args.verbose)
        sys.exit(0 if fixed.ok else 1)
    except (ValueError, ET.ParseError, OSError) as e:
        print(error(f"ERROR: {e}"))
        sys.exit(2)


if __name__ == "__main__":
    main()
