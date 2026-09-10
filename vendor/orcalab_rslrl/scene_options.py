"""Canonical scene-level physics profiles for training and runtime playback."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco


@dataclass(frozen=True)
class UnitreeOrcaSceneOptions:
    """Scene-wide values exported by unitree-orca after runtime overrides."""

    timestep: float = 0.001
    integrator: int = int(mujoco.mjtIntegrator.mjINT_EULER)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    density: float = 0.0
    viscosity: float = 0.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    iterations: int = 100
    ls_iterations: int = 50
    noslip_iterations: int = 0
    ccd_iterations: int = 35
    sdf_initpoints: int = 40
    sdf_iterations: int = 10
    tolerance: float = 1.0e-8
    ls_tolerance: float = 1.0e-2
    noslip_tolerance: float = 1.0e-6
    ccd_tolerance: float = 1.0e-6


@dataclass(frozen=True)
class OrcaTrainSceneOptions:
    """Canonical physics options for Orca locomotion training."""

    timestep: float = 0.005
    integrator: int = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    density: float = 0.0
    viscosity: float = 0.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    iterations: int = 10
    ls_iterations: int = 20
    noslip_iterations: int = 0
    ccd_iterations: int = 50
    sdf_initpoints: int = 40
    sdf_iterations: int = 10
    tolerance: float = 1.0e-8
    ls_tolerance: float = 1.0e-2
    noslip_tolerance: float = 1.0e-6
    ccd_tolerance: float = 1.0e-6


UNITREE_ORCA_SCENE_OPTIONS = UnitreeOrcaSceneOptions()
ORCA_TRAIN_SCENE_OPTIONS = OrcaTrainSceneOptions()
UNITREE_ORCA_CONTROL_DT = 0.02
UNITREE_ORCA_GROUND_FRICTION = (1.0, 0.005, 0.0001)
UNITREE_ORCA_GROUND_SOLREF = (0.02, 1.0)
UNITREE_ORCA_GROUND_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

SCENE_OPTION_PROFILES = {
    "orca-runtime": UNITREE_ORCA_SCENE_OPTIONS,
    "orca-train": ORCA_TRAIN_SCENE_OPTIONS,
}


def scene_xml_contract(path: str | Path) -> dict[str, Any]:
    """Inspect the source/effective XML without compiling or mutating it."""

    xml_path = Path(path).expanduser().resolve()
    payload = xml_path.read_bytes()
    root = ET.fromstring(payload)
    option = root.find("option")
    ground_geoms = []
    for geom in root.iter("geom"):
        if geom.get("type", "").lower() != "plane":
            continue
        ground_geoms.append(
            {
                "name": geom.get("name", ""),
                "friction": geom.get("friction"),
                "solref": geom.get("solref"),
                "solimp": geom.get("solimp"),
                "contype": geom.get("contype"),
                "conaffinity": geom.get("conaffinity"),
                "condim": geom.get("condim"),
            }
        )
    return {
        "path": str(xml_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "option_attributes": {} if option is None else dict(option.attrib),
        "ground_geoms": ground_geoms,
    }


def resolve_scene_options_profile(
    profile: str,
) -> UnitreeOrcaSceneOptions | OrcaTrainSceneOptions:
    """Resolve a named, versioned scene profile.

    ``orca-train`` is the profile used by the Unitree G1 flat task:
    5 ms physics, ImplicitFast and the 10/20/50 solver iteration tuple.
    """

    try:
        return SCENE_OPTION_PROFILES[str(profile)]
    except KeyError as exc:
        choices = ", ".join(sorted(SCENE_OPTION_PROFILES))
        raise ValueError(
            f"Unknown scene option profile {profile!r}; choose one of: {choices}"
        ) from exc


def scene_options_snapshot(
    target,
    *,
    profile: str,
) -> dict[str, Any]:
    """Return the effective MuJoCo options as a JSON-serializable contract."""

    opt = target.option if hasattr(target, "option") else target.opt
    return {
        "profile": str(profile),
        "timestep": float(opt.timestep),
        "integrator": int(opt.integrator),
        "gravity": [float(value) for value in opt.gravity],
        "density": float(opt.density),
        "viscosity": float(opt.viscosity),
        "wind": [float(value) for value in opt.wind],
        "iterations": int(opt.iterations),
        "ls_iterations": int(opt.ls_iterations),
        "noslip_iterations": int(opt.noslip_iterations),
        "ccd_iterations": int(opt.ccd_iterations),
        "sdf_initpoints": int(opt.sdf_initpoints),
        "sdf_iterations": int(opt.sdf_iterations),
        "tolerance": float(opt.tolerance),
        "ls_tolerance": float(opt.ls_tolerance),
        "noslip_tolerance": float(opt.noslip_tolerance),
        "ccd_tolerance": float(opt.ccd_tolerance),
    }


def assert_scene_options(
    target,
    *,
    profile: str,
    timestep: float | None = None,
    align_air_resistance: bool = True,
    atol: float = 1.0e-9,
) -> dict[str, Any]:
    """Fail fast unless a compiled/downloaded model matches the requested profile."""

    options = resolve_scene_options_profile(profile)
    actual = scene_options_snapshot(target, profile=profile)
    expected_timestep = options.timestep if timestep is None else float(timestep)
    expected: dict[str, Any] = {
        "timestep": expected_timestep,
        "integrator": options.integrator,
        "gravity": list(options.gravity),
        "iterations": options.iterations,
        "ls_iterations": options.ls_iterations,
        "noslip_iterations": options.noslip_iterations,
        "ccd_iterations": options.ccd_iterations,
        "sdf_initpoints": options.sdf_initpoints,
        "sdf_iterations": options.sdf_iterations,
        "tolerance": options.tolerance,
        "ls_tolerance": options.ls_tolerance,
        "noslip_tolerance": options.noslip_tolerance,
        "ccd_tolerance": options.ccd_tolerance,
    }
    if align_air_resistance:
        expected.update(
            {
                "density": options.density,
                "viscosity": options.viscosity,
                "wind": list(options.wind),
            }
        )

    mismatches = []
    for key, wanted in expected.items():
        got = actual[key]
        if isinstance(wanted, list):
            if len(got) != len(wanted) or any(
                abs(float(left) - float(right)) > atol
                for left, right in zip(got, wanted, strict=True)
            ):
                mismatches.append(f"{key}={got!r} expected {wanted!r}")
        elif isinstance(wanted, float):
            if abs(float(got) - wanted) > atol:
                mismatches.append(f"{key}={got!r} expected {wanted!r}")
        elif got != wanted:
            mismatches.append(f"{key}={got!r} expected {wanted!r}")
    if mismatches:
        raise RuntimeError(
            f"MuJoCo scene profile {profile!r} is not aligned: " + "; ".join(mismatches)
        )
    return actual


def assert_flat_ground_options(
    model: mujoco.MjModel,
    *,
    atol: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Verify every plane geom against the flat locomotion contact contract.

    OrcaLab scene values can make a float32 round trip before they are loaded
    into MuJoCo, so the comparison tolerates representation noise while still
    rejecting meaningful contact-profile changes.
    """

    planes: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        snapshot = {
            "id": geom_id,
            "name": name,
            "friction": model.geom_friction[geom_id].tolist(),
            "solref": model.geom_solref[geom_id].tolist(),
            "solimp": model.geom_solimp[geom_id].tolist(),
            "contype": int(model.geom_contype[geom_id]),
            "conaffinity": int(model.geom_conaffinity[geom_id]),
            "condim": int(model.geom_condim[geom_id]),
        }
        planes.append(snapshot)
        vector_expected = {
            "friction": UNITREE_ORCA_GROUND_FRICTION,
            "solref": UNITREE_ORCA_GROUND_SOLREF,
            "solimp": UNITREE_ORCA_GROUND_SOLIMP,
        }
        for key, wanted in vector_expected.items():
            if not np_allclose(snapshot[key], wanted, atol=atol):
                mismatches.append(
                    f"{name or geom_id}.{key}={snapshot[key]!r} expected {list(wanted)!r}"
                )
        for key, wanted in {"contype": 1, "conaffinity": 1, "condim": 3}.items():
            if snapshot[key] != wanted:
                mismatches.append(
                    f"{name or geom_id}.{key}={snapshot[key]!r} expected {wanted!r}"
                )
    if not planes:
        raise RuntimeError("aligned OrcaLab XML has no flat ground plane geom")
    if mismatches:
        raise RuntimeError(
            "flat ground contact profile is not aligned: " + "; ".join(mismatches)
        )
    return planes


def np_allclose(
    actual: list[float],
    expected: tuple[float, ...],
    *,
    atol: float,
) -> bool:
    """Small NumPy-free vector comparison for scene-profile validation."""

    return len(actual) == len(expected) and all(
        abs(float(left) - float(right)) <= atol
        for left, right in zip(actual, expected, strict=True)
    )


def patch_scene_xml_options(
    source_path: str | Path,
    output_path: str | Path,
    *,
    profile: str,
    timestep: float | None = None,
    align_air_resistance: bool = True,
) -> Path:
    """Write a non-destructive XML copy with the selected scene options.

    OrcaLab's downloaded combined XML is retained as the source of scene bodies,
    geoms and actor prefixes.  The global ``<option>`` and flat-ground contact
    contract are overwritten; the original downloaded file is never modified
    in place.
    """

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source == output:
        raise ValueError(
            "aligned scene XML output must not overwrite the OrcaLab source XML"
        )
    options = resolve_scene_options_profile(profile)
    tree = ET.parse(source)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)

    resolved_timestep = options.timestep if timestep is None else float(timestep)
    integrator_names = {
        int(mujoco.mjtIntegrator.mjINT_EULER): "Euler",
        int(mujoco.mjtIntegrator.mjINT_IMPLICIT): "implicit",
        int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast",
        int(mujoco.mjtIntegrator.mjINT_RK4): "RK4",
    }
    option_values: dict[str, str] = {
        "timestep": f"{resolved_timestep:.17g}",
        "integrator": integrator_names[int(options.integrator)],
        "gravity": " ".join(f"{value:.17g}" for value in options.gravity),
        "iterations": str(options.iterations),
        "ls_iterations": str(options.ls_iterations),
        "noslip_iterations": str(options.noslip_iterations),
        "ccd_iterations": str(options.ccd_iterations),
        "sdf_initpoints": str(options.sdf_initpoints),
        "sdf_iterations": str(options.sdf_iterations),
        "tolerance": f"{options.tolerance:.17g}",
        "ls_tolerance": f"{options.ls_tolerance:.17g}",
        "noslip_tolerance": f"{options.noslip_tolerance:.17g}",
        "ccd_tolerance": f"{options.ccd_tolerance:.17g}",
    }
    if align_air_resistance:
        option_values.update(
            {
                "density": f"{options.density:.17g}",
                "viscosity": f"{options.viscosity:.17g}",
                "wind": " ".join(f"{value:.17g}" for value in options.wind),
            }
        )
    for name, value in option_values.items():
        option.set(name, value)

    ground_values = {
        "friction": " ".join(f"{value:.17g}" for value in UNITREE_ORCA_GROUND_FRICTION),
        "solref": " ".join(f"{value:.17g}" for value in UNITREE_ORCA_GROUND_SOLREF),
        "solimp": " ".join(f"{value:.17g}" for value in UNITREE_ORCA_GROUND_SOLIMP),
        "contype": "1",
        "conaffinity": "1",
        "condim": "3",
    }
    for geom in root.iter("geom"):
        if geom.get("type", "").lower() != "plane":
            continue
        for name, value in ground_values.items():
            geom.set(name, value)

    # OrcaLab writes the combined XML beside generated mesh/texture/include
    # files.  The aligned artifact lives in the NaVILA output directory, so
    # make those references source-relative before moving the XML; otherwise
    # an option-only repair can accidentally make a valid scene unloadable.
    compiler = root.find("compiler")
    source_dir = source.parent
    asset_dir: Path | None = None
    mesh_dir: Path | None = None
    texture_dir: Path | None = None
    if compiler is not None:
        resolved_dirs: dict[str, Path] = {}
        for attribute in ("assetdir", "meshdir", "texturedir"):
            raw_value = compiler.get(attribute)
            if not raw_value:
                continue
            path = Path(raw_value).expanduser()
            if not path.is_absolute():
                path = source_dir / path
            path = path.resolve()
            compiler.set(attribute, str(path))
            resolved_dirs[attribute] = path
        asset_dir = resolved_dirs.get("assetdir")
        mesh_dir = resolved_dirs.get("meshdir", asset_dir)
        texture_dir = resolved_dirs.get("texturedir", asset_dir)

    for element in root.iter():
        raw_file = element.get("file")
        if not raw_file:
            continue
        file_path = Path(raw_file).expanduser()
        if file_path.is_absolute():
            continue
        tag = element.tag.rsplit("}", 1)[-1]
        has_external_base = (tag == "mesh" and mesh_dir is not None) or (
            tag == "texture" and texture_dir is not None
        )
        if not has_external_base:
            element.set("file", str((source_dir / file_path).resolve()))

    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output


def apply_scene_options(
    target,
    *,
    options: UnitreeOrcaSceneOptions | OrcaTrainSceneOptions,
    timestep: float | None = None,
    align_air_resistance: bool = True,
) -> None:
    """Apply a scene preset to an ``MjSpec`` or compiled ``MjModel``."""

    opt = target.option if hasattr(target, "option") else target.opt
    resolved_timestep = options.timestep if timestep is None else float(timestep)
    if resolved_timestep <= 0.0:
        raise ValueError(f"timestep must be positive, got {resolved_timestep}")

    opt.timestep = resolved_timestep
    opt.integrator = options.integrator
    opt.gravity[:] = options.gravity
    opt.iterations = options.iterations
    opt.ls_iterations = options.ls_iterations
    opt.noslip_iterations = options.noslip_iterations
    opt.ccd_iterations = options.ccd_iterations
    opt.sdf_initpoints = options.sdf_initpoints
    opt.sdf_iterations = options.sdf_iterations
    opt.tolerance = options.tolerance
    opt.ls_tolerance = options.ls_tolerance
    opt.noslip_tolerance = options.noslip_tolerance
    opt.ccd_tolerance = options.ccd_tolerance
    if align_air_resistance:
        opt.density = options.density
        opt.viscosity = options.viscosity
        opt.wind[:] = options.wind


def apply_unitree_orca_scene_options(
    target,
    *,
    timestep: float | None = None,
    align_air_resistance: bool = True,
) -> None:
    """Apply the unitree-orca runtime playback preset."""

    apply_scene_options(
        target,
        options=UNITREE_ORCA_SCENE_OPTIONS,
        timestep=timestep,
        align_air_resistance=align_air_resistance,
    )


def apply_orca_train_scene_options(
    target,
    *,
    timestep: float | None = None,
) -> None:
    """Apply the Unitree Orca flat-task training preset."""

    apply_scene_options(
        target,
        options=ORCA_TRAIN_SCENE_OPTIONS,
        timestep=timestep,
    )
