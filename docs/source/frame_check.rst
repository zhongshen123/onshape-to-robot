.. _frame_check:

Checking and aligning frames
============================

Convention
----------

Many tools expect every link frame to have the same orientation as the robot base at the zero
configuration: **x forward, y left, z up**, with every joint axis along ``±X``, ``±Y`` or ``±Z``.

``onshape-to-robot`` uses each joint's mate connector as the child link frame, so link frames follow
however the mate connectors were placed in Onshape. The tools below check this convention and fix it.

The reference frame is the root link (URDF) or the world (MuJoCo), which is the Onshape **assembly
origin**. Orient the robot in Onshape so it faces +X with Z up.

Checking an exported file
-------------------------

.. code-block:: bash

    onshape-to-robot-check my-robot/robot.urdf     # or robot.xml, or the my-robot/ directory

The command lists every link frame that is rotated relative to the world (and where its x/y/z axes
point), and every joint axis that is not along ``±X/±Y/±Z``. It exits with code ``1`` if anything is
wrong, so it can be used in scripts or CI.

Options:

* ``--fix``: write ``robot_aligned.urdf`` / ``robot_aligned.xml`` where misaligned link frames are
  re-oriented. Geometry, inertia, joint axes and frames stay exactly where they were in the world;
  joint axes are re-expressed (e.g. ``0 0 1`` may become ``0 -1 0``). Use ``-o FILE`` to choose the
  output, or ``--in-place`` to overwrite.
* ``--tolerance DEG``: angle tolerance (default ``0.5``). Frames within the tolerance are left untouched
  by ``--fix``; use ``--tolerance 0`` to align everything exactly.
* ``--include-frames``: also check (and align) ``frame_*`` / ``closing_*`` frames and MuJoCo sites.
  By default they keep their own orientation (e.g. a camera frame).
* ``--ignore PATTERN ...``: skip links/joints matching these patterns.
* ``-v``: print the world direction of every joint axis.

A tilted joint axis (not along ``±X/±Y/±Z`` in the world) cannot be fixed by re-orienting frames: fix the
mate in Onshape.

``config.json`` entries
-----------------------

.. code-block:: javascript

    {
        // Align every link frame with the world at export time
        "align_frames": true,

        // Run the check on the output after every export
        "check_frames": true
    }

``align_frames`` *(default: false)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``true``, every link frame is made world-aligned during export (all formats). Joint axes, including
the ``xyz`` joint property, are converted accordingly, so the robot geometry and motion do not change.
Can also be a list of joint name patterns, to align only the child links of these joints.

``check_frames`` *(default: false)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``true``, the frame check runs on the exported URDF or MuJoCo file and its report is printed. The export
does not fail when the check fails.
