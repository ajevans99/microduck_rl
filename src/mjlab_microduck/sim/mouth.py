"""Hinged lower beak (``mouth`` joint + actuator) injection for MuJoCo simulation.

The Onshape-exported ``robot_*.xml`` models merge the lower beak (``jaw.stl`` +
``jaw_soft.stl``) rigidly into the head body (``jaw_soft``) so all RL training
environments, ``scripts/infer_policy.py``, and 14-DOF scene keyframes stay
14-action / 61D-obs without needing every task config to filter out a 15th joint.

When serving a simulated body to ``robotd`` (``sim/body_server.py``), ``robotd``
speaks the full 15-joint wire protocol including ``mouth`` at index 9
(``-5 deg`` closed to ``+30 deg`` open). ``add_hinged_mouth(spec)`` splits the
lower beak geoms out of the head body into a hinged child ``jaw`` body on the
``MjSpec`` before compilation, conserving the composite head mass, CoM, and
inertia tensor and keeping existing 14-DOF scene keyframes valid.

Idempotent: if a future CAD export already defines a ``mouth`` joint or
actuator, this is a no-op.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

MOUTH_JOINT_NAME = "mouth"
JAW_BODY_NAME = "jaw"
JAW_MESH_NAMES = ("jaw", "jaw_soft")

# Hardware travel limits matching ``duck-control/src/model.rs``:
#   MOUTH_CLOSED = -5 deg, MOUTH_OPEN = +30 deg.
MOUTH_CLOSED_RAD = math.radians(-5.0)
MOUTH_OPEN_RAD = math.radians(30.0)

# Exact hinge axis in the head (``jaw_soft``) body frame: coaxial with the
# mouth servo horn (``xl330_10`` at [0.003, 0.0255, -0.018]) and opposing
# bearing (``seeed_bearing__configuration_default_2`` at [0.0032, -0.04, -0.018]),
# verified against the circular r=8/9 mm hinge ears of ``jaw.stl``.
JAW_PIVOT_IN_HEAD = np.array([0.003, 0.0, -0.018], dtype=np.float64)

# Local +Y of ``jaw_soft`` points left in the robot frame; by the right-hand
# rule, positive rotation (+30 deg) swings the lower beak downward (open) and
# negative rotation (-5 deg) closes it snugly against ``soft_mouth_top``.
JAW_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)

# Mass, CoM (in ``jaw_soft`` frame), and inertia about CoM (in ``jaw_soft``
# axes) computed from ``jaw.stl`` (9.360 cm^3) + ``jaw_soft.stl`` (18.601 cm^3)
# at 1000 kg/m^3 nominal printed PLA/TPU density (matching the Onshape plastic
# link densities in ``robot_*.xml``). Subtracted from the parent head body via
# the parallel-axis theorem so composite head mass/CoM/inertia at mouth=0 are
# preserved exactly.
JAW_MASS = 0.02796132
JAW_COM_IN_HEAD = np.array(
    [-0.011239556836123717, -0.0005636606780535535, -0.05706212100266368],
    dtype=np.float64,
)
JAW_FULLINERTIA = np.array(
    [
        2.8890527448455544e-05,
        7.038925182718702e-06,
        2.3150612195041006e-05,
        2.2526280676431548e-07,
        -1.1555208977605247e-06,
        6.15400854402432e-07,
    ],
    dtype=np.float64,
)

# Fallback XL330 PD parameters (matching ``chosen_actuator`` in
# ``joints_properties.xml``) when a spec has no existing servo actuators.
_FALLBACK_KP = 0.55
_FALLBACK_FORCERANGE = (-0.96, 0.96)
_FALLBACK_CTRLRANGE = (-10.0, 10.0)
_FALLBACK_DAMPING = 0.053
_FALLBACK_FRICTIONLOSS = 0.0048
_FALLBACK_ARMATURE = 0.0018

_QPOS_DIM = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


def _fullinertia_to_mat(fi: np.ndarray) -> np.ndarray:
    ixx, iyy, izz, ixy, ixz, iyz = fi
    return np.array(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64
    )


def _mat_to_fullinertia(mat: np.ndarray) -> np.ndarray:
    return np.array(
        [mat[0, 0], mat[1, 1], mat[2, 2], mat[0, 1], mat[0, 2], mat[1, 2]],
        dtype=np.float64,
    )


def _shift_inertia(inertia_com: np.ndarray, mass: float, offset: np.ndarray) -> np.ndarray:
    return inertia_com + mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))


def _find_head_body(spec: mujoco.MjSpec):
    head = spec.body("jaw_soft")
    if head is not None and any(g.meshname in JAW_MESH_NAMES for g in head.geoms):
        return head
    # Fallback for CAD exports where the merged head link took another part name.
    for body in spec.bodies:
        if any(g.meshname == "jaw" for g in body.geoms):
            return body
    return None


def _dfs_qpos_offset_for_new_child(spec: mujoco.MjSpec, parent_body) -> int | None:
    """Return the qpos index a new child body appended to ``parent_body`` will get."""
    count = 0
    found = False

    def walk(body) -> None:
        nonlocal count, found
        if found:
            return
        for joint in body.joints:
            count += _QPOS_DIM[joint.type]
        for child in body.bodies:
            walk(child)
            if found:
                return
        if body is parent_body:
            found = True

    walk(spec.worldbody)
    return count if found else None


def _subtract_jaw_inertia(head_body) -> None:
    """Subtract the split-out jaw inertia from the parent head body if valid."""
    total_mass = float(head_body.mass)
    if total_mass <= JAW_MASS * 1.5:
        return
    c_tot = np.asarray(head_body.ipos, dtype=np.float64)
    fi_tot = np.asarray(head_body.fullinertia, dtype=np.float64)
    if np.all(fi_tot == 0.0) or not np.all(np.isfinite(fi_tot)):
        return

    i_tot = _fullinertia_to_mat(fi_tot)
    i_jaw = _fullinertia_to_mat(JAW_FULLINERTIA)
    m_rem = total_mass - JAW_MASS
    c_rem = (total_mass * c_tot - JAW_MASS * JAW_COM_IN_HEAD) / m_rem

    i_jaw_at_ctot = _shift_inertia(i_jaw, JAW_MASS, c_tot - JAW_COM_IN_HEAD)
    i_rem_at_ctot = i_tot - i_jaw_at_ctot
    i_rem = _shift_inertia(i_rem_at_ctot, -m_rem, c_tot - c_rem)

    # Keep the parent inertia valid only if subtraction remains strictly positive-definite.
    if np.all(np.linalg.eigvalsh(i_rem) > 1e-9):
        head_body.mass = m_rem
        head_body.ipos = c_rem
        head_body.fullinertia = _mat_to_fullinertia(i_rem)


def add_hinged_mouth(spec: mujoco.MjSpec) -> bool:
    """Split ``jaw`` + ``jaw_soft`` geoms into an actuated hinged ``mouth`` body.

    Returns ``True`` if the hinged mouth was added, or ``False`` if ``spec``
    already has a ``mouth`` joint/actuator or has no jaw geometry to split.
    """
    if any(j.name == MOUTH_JOINT_NAME for j in spec.joints):
        return False
    if any(a.name == MOUTH_JOINT_NAME for a in spec.actuators):
        return False

    head = _find_head_body(spec)
    if head is None:
        return False

    jaw_geoms = [g for g in list(head.geoms) if g.meshname in JAW_MESH_NAMES]
    if not jaw_geoms:
        return False

    ins_qpos = _dfs_qpos_offset_for_new_child(spec, head)

    # Inherit the servo default class (``chosen_actuator``) when present so the
    # mouth joint/actuator matches the other XL330 servos and ``Body._apply_torque``.
    template_act = spec.actuators[0] if len(spec.actuators) > 0 else None
    servo_default = template_act.classname if template_act is not None else None

    _subtract_jaw_inertia(head)

    jaw_body = head.add_body(name=JAW_BODY_NAME, pos=JAW_PIVOT_IN_HEAD)
    jaw_body.mass = JAW_MASS
    jaw_body.ipos = JAW_COM_IN_HEAD - JAW_PIVOT_IN_HEAD
    jaw_body.fullinertia = JAW_FULLINERTIA

    joint_kwargs = dict(
        name=MOUTH_JOINT_NAME,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=JAW_AXIS,
        limited=True,
        range=[MOUTH_CLOSED_RAD, MOUTH_OPEN_RAD],
    )
    if servo_default is not None:
        jaw_body.add_joint(default=servo_default, **joint_kwargs)
    else:
        jaw_body.add_joint(
            damping=_FALLBACK_DAMPING,
            frictionloss=_FALLBACK_FRICTIONLOSS,
            armature=_FALLBACK_ARMATURE,
            **joint_kwargs,
        )

    for geom in jaw_geoms:
        geom_kwargs = dict(
            type=geom.type,
            meshname=geom.meshname,
            material=geom.material,
            pos=np.asarray(geom.pos, dtype=np.float64) - JAW_PIVOT_IN_HEAD,
            quat=geom.quat,
        )
        if geom.classname is not None:
            jaw_body.add_geom(default=geom.classname, **geom_kwargs)
        else:
            jaw_body.add_geom(**geom_kwargs)
        spec.delete(geom)

    # Preserve the contact exclusions the jaw geoms previously inherited from
    # being part of ``jaw_soft`` (direct child of ``yaw_roll_motion``, excluded
    # from ``neck_pitch`` in ``allcollisions_contacts.xml``).
    spec.add_exclude(bodyname1=JAW_BODY_NAME, bodyname2=head.name)
    for neighbor in ("yaw_roll_motion", "neck_pitch"):
        if spec.body(neighbor) is not None:
            spec.add_exclude(bodyname1=neighbor, bodyname2=JAW_BODY_NAME)

    if servo_default is not None and template_act is not None:
        act = spec.add_actuator(
            default=servo_default,
            name=MOUTH_JOINT_NAME,
            target=MOUTH_JOINT_NAME,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )
        act.gaintype = template_act.gaintype
        act.biastype = template_act.biastype
        act.gainprm = template_act.gainprm
        act.biasprm = template_act.biasprm
    else:
        act = spec.add_actuator(
            name=MOUTH_JOINT_NAME,
            target=MOUTH_JOINT_NAME,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = _FALLBACK_KP
        act.biasprm[1] = -_FALLBACK_KP
        act.forcelimited = True
        act.forcerange = _FALLBACK_FORCERANGE
        act.ctrllimited = True
        act.ctrlrange = _FALLBACK_CTRLRANGE

    # Keep existing 14-DOF scene keyframes (built-in and external) compilable
    # by inserting 0.0 at the new mouth qpos/ctrl slots.
    for key in spec.keys:
        if ins_qpos is not None and len(key.qpos) >= ins_qpos and len(key.qpos) > 0:
            key.qpos = np.insert(np.asarray(key.qpos, dtype=np.float64), ins_qpos, 0.0)
        if len(key.ctrl) > 0:
            key.ctrl = np.append(np.asarray(key.ctrl, dtype=np.float64), 0.0)

    return True
