"""Tests for hinged lower beak (``mouth``) injection in ``sim/mouth.py`` and ``sim/body_server.py``."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.body_server import (
    DEFAULT_SCENE,
    HOME_POSE,
    JOINT_NAMES,
    MOUTH_INDEX,
    SCENES,
    Body,
    World,
    build_world,
    pose_table,
)
from mjlab_microduck.sim.mouth import (
    JAW_BODY_NAME,
    JAW_MASS,
    MOUTH_CLOSED_RAD,
    MOUTH_JOINT_NAME,
    MOUTH_OPEN_RAD,
    add_hinged_mouth,
)

ALL_SCENES = sorted(SCENES.glob("scene*.xml"))


@pytest.mark.parametrize("scene", ALL_SCENES, ids=lambda p: p.name)
def test_all_scenes_compile_with_hinged_mouth(scene: Path):
    base = mujoco.MjModel.from_xml_path(str(scene))
    model = build_world(scene, count=1)

    assert model.nq == base.nq + 1
    assert model.nu == base.nu + 1
    assert model.nkey == base.nkey

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MOUTH_JOINT_NAME)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MOUTH_JOINT_NAME)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, JAW_BODY_NAME)
    assert jid >= 0 and aid >= 0 and bid >= 0
    assert model.jnt_limited[jid] == 1
    assert np.allclose(model.jnt_range[jid], [MOUTH_CLOSED_RAD, MOUTH_OPEN_RAD])
    assert model.actuator_trnid[aid, 0] == jid

    # Keyframes must preserve every named original joint pose with mouth=0.
    for k in range(base.nkey):
        for j in range(base.njnt):
            jname = mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jname is None:
                continue
            new_j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            b_adr = base.jnt_qposadr[j]
            n_adr = model.jnt_qposadr[new_j]
            size = 7 if base.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
            assert np.allclose(
                model.key_qpos[k, n_adr : n_adr + size],
                base.key_qpos[k, b_adr : b_adr + size],
            )
        assert model.key_qpos[k, model.jnt_qposadr[jid]] == pytest.approx(0.0)


def test_head_mass_com_and_inertia_conserved():
    base = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    hinged = build_world(DEFAULT_SCENE, count=1)

    base_data = mujoco.MjData(base)
    hinged_data = mujoco.MjData(hinged)
    mujoco.mj_forward(base, base_data)
    mujoco.mj_forward(hinged, hinged_data)

    # Total robot mass and head subtree mass/CoM must be identical at mouth=0.
    assert hinged.body_subtreemass[0] == pytest.approx(base.body_subtreemass[0], abs=1e-9)

    b_head = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    h_head = mujoco.mj_name2id(hinged, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    h_jaw = mujoco.mj_name2id(hinged, mujoco.mjtObj.mjOBJ_BODY, JAW_BODY_NAME)
    assert hinged.body_mass[h_jaw] == pytest.approx(JAW_MASS, rel=1e-6)
    assert hinged.body_subtreemass[h_head] == pytest.approx(base.body_subtreemass[b_head], abs=1e-9)
    assert np.allclose(hinged_data.subtree_com[h_head], base_data.subtree_com[b_head], atol=1e-7)


def test_no_phantom_self_contacts_on_allcollisions():
    scene = SCENES / "scene_allcollisions.xml"
    model = build_world(scene, count=1)
    data = mujoco.MjData(model)
    jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, JAW_BODY_NAME)
    mjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MOUTH_JOINT_NAME)

    for k in range(model.nkey):
        for angle in (MOUTH_CLOSED_RAD, 0.0, MOUTH_OPEN_RAD):
            mujoco.mj_resetDataKeyframe(model, data, k)
            data.qpos[model.jnt_qposadr[mjid]] = angle
            mujoco.mj_forward(model, data)
            for ci in range(data.ncon):
                c = data.contact[ci]
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                if jaw_bid in (b1, b2):
                    other = b2 if b1 == jaw_bid else b1
                    other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
                    assert other_name not in ("jaw_soft", "yaw_roll_motion", "neck_pitch")


def test_idempotent_on_existing_mouth():
    spec = mujoco.MjSpec.from_file(str(DEFAULT_SCENE))
    assert add_hinged_mouth(spec) is True
    assert add_hinged_mouth(spec) is False
    model = spec.compile()
    assert sum(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) == MOUTH_JOINT_NAME
        for j in range(model.njnt)
    ) == 1


def test_body_drives_mouth_closed_and_open_and_multiduck():
    world = World(DEFAULT_SCENE, count=2)
    pose, trunk_z = pose_table(DEFAULT_SCENE, "SIT")
    ducks = []
    for i in range(2):
        body = Body(world, i, limp=False)
        assert len(body.actuators) == len(JOINT_NAMES) == 15
        assert MOUTH_INDEX in body.to_wire
        body.place(pose, trunk_z, offset_y=i * 0.5)
        world.bodies.append(body)
        ducks.append(body)

    for b in ducks:
        b.set_torque(True)

    # Command duck 0 fully open (+30 deg) and duck 1 fully closed (-5 deg).
    t0 = list(HOME_POSE)
    t0[MOUTH_INDEX] = MOUTH_OPEN_RAD
    t1 = list(HOME_POSE)
    t1[MOUTH_INDEX] = MOUTH_CLOSED_RAD
    ducks[0].set_targets(t0)
    ducks[1].set_targets(t1)

    # Hold trunks in place by stepping 1.0 s (200 steps at 5 ms) and checking jaw convergence.
    world.step(200)
    s0 = ducks[0].sensors()
    s1 = ducks[1].sensors()

    assert s0["positions"][MOUTH_INDEX] == pytest.approx(MOUTH_OPEN_RAD, abs=0.05)
    assert s1["positions"][MOUTH_INDEX] == pytest.approx(MOUTH_CLOSED_RAD, abs=0.05)

    # Verify positive mouth angle rotates the jaw CoM downward in the head frame
    # (jaw_soft local +X points up in the head frame regardless of trunk tilt).
    jaw0 = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, JAW_BODY_NAME)
    jaw1 = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, f"d1_{JAW_BODY_NAME}")
    head0 = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    head1 = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "d1_jaw_soft")
    head0_up = world.data.xmat[head0].reshape(3, 3)[:, 0]
    head1_up = world.data.xmat[head1].reshape(3, 3)[:, 0]
    up_open = float(np.dot(world.data.xipos[jaw0] - world.data.xpos[head0], head0_up))
    up_closed = float(np.dot(world.data.xipos[jaw1] - world.data.xpos[head1], head1_up))
    assert up_open < up_closed - 0.020


def test_external_backlash_scene_without_mouth_keyframe(tmp_path: Path):
    """Simulate an external scene (e.g. MicroduckRemote LiDAR export) including robot_groundcontact_backlash.xml."""
    robot_xml = (SCENES / "robot_groundcontact_backlash.xml").resolve()
    assets_dir = (SCENES / "assets").resolve()
    ext_scene = tmp_path / "lidar_room.xml"
    ext_scene.write_text(
        f"""<mujoco model="lidar_room">
  <include file="{robot_xml}"/>
  <compiler meshdir="{assets_dir}"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
  </worldbody>
  <keyframe>
    <key name="INIT" qpos="{' '.join(['0']*35)}" ctrl="{' '.join(['0']*14)}"/>
  </keyframe>
</mujoco>
"""
    )
    model = build_world(ext_scene, count=1)
    assert model.nq == 36
    assert model.nu == 15
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MOUTH_JOINT_NAME) >= 0
