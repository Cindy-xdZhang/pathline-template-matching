from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import pathline_template_matching.seed_time_kinematic_sidecar as sidecar_module

from pathline_template_matching.arc_length_primitives import (
    _interp4_quadrilinear_scalar,
)
from pathline_template_matching.early_opposite_pair_kinematics import (
    FROZEN_PRIMITIVE_ORDER,
    compute_seed_time_velocity_gradient,
)
from pathline_template_matching.netcdf_io import FlowWindow3D
from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    sha256_file,
)
from pathline_template_matching.seed_time_kinematic_sidecar import (
    FORBIDDEN_PARENT_MEMBER_NAMES,
    FROZEN_DX_GRID_SCALE_BY_ID,
    PARENT_PROJECTION_MEMBER_NAMES,
    SIDECAR_ARCHIVE_MEMBER_NAMES,
    SIDECAR_ARRAY_NAMES,
    SIDECAR_PROVENANCE_BINDING_NAMES,
    ParentKinematicProjection,
    SeedTimeKinematicSidecarPayload,
    build_seed_time_kinematic_sidecar_payload,
    load_parent_kinematic_projection,
    load_seed_time_kinematic_sidecar,
    physical_dx_by_scale_for_window,
    sample_seed_time_velocity_xyz,
    validate_sidecar_identity_join,
    write_seed_time_kinematic_sidecar,
)
from pathline_template_matching.vector_field import UnsteadyVectorField3D


_OFFSETS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _expect_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _affine_window(
    matrix: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    *,
    frame_one_offset: float = 1000.0,
) -> FlowWindow3D:
    if matrix is None:
        matrix = np.asarray(
            [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 1.0, 1.25]],
            dtype=np.float64,
        )
    if translation is None:
        translation = np.asarray([0.25, -0.5, 1.0], dtype=np.float64)
    x = np.linspace(-1.0, 1.0, 17, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, 17, dtype=np.float64)
    z = np.linspace(-1.0, 1.0, 17, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    positions = np.stack((xx, yy, zz), axis=-1)
    first = positions @ np.asarray(matrix, dtype=np.float64).T
    first += np.asarray(translation, dtype=np.float64)
    velocity = np.stack((first, first + frame_one_offset), axis=0).astype(np.float32)
    return FlowWindow3D(
        velocity=velocity,
        coordinates_xyz=(x, y, z),
        time=np.asarray([17.25, 17.75], dtype=np.float64),
        source_path="synthetic-only.nc",
        source_start_index=0,
        spatial_strides={"x": 1, "y": 1, "z": 1},
        components=("u", "v", "w"),
        coordinate_sources={"x": "synthetic", "y": "synthetic", "z": "synthetic", "t": "synthetic"},
    )


def _nonlinear_window() -> FlowWindow3D:
    x = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    y = np.linspace(-0.75, 0.75, 7, dtype=np.float64)
    z = np.linspace(-0.5, 0.5, 5, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    first = np.stack(
        (
            0.25 + xx * yy + 0.1 * zz,
            -0.5 + yy * zz - 0.2 * xx,
            1.0 + zz * xx + 0.3 * yy,
        ),
        axis=-1,
    )
    second = first * -7.0 + 123.0
    return FlowWindow3D(
        velocity=np.stack((first, second), axis=0).astype(np.float32),
        coordinates_xyz=(x, y, z),
        time=np.asarray([8.0, 8.25], dtype=np.float64),
        source_path="synthetic-nonlinear.nc",
        source_start_index=3,
        spatial_strides={"x": 1, "y": 1, "z": 1},
        components=("u", "v", "w"),
        coordinate_sources={"x": "synthetic", "y": "synthetic", "z": "synthetic", "t": "synthetic"},
    )


def _parent_arrays(scale_ids: np.ndarray) -> dict[str, np.ndarray]:
    scale = np.asarray(scale_ids, dtype=np.int32)
    if np.any(np.diff((scale >= 1000).astype(np.int8)) < 0):
        raise ValueError("test scale ids must list legacy rows before expanded rows")
    legacy_count = int(np.sum(scale < 1000))
    expanded_count = len(scale) - legacy_count
    center_count = max(legacy_count, expanded_count, 1)
    center = np.concatenate(
        (
            np.arange(legacy_count, dtype=np.int64),
            np.arange(expanded_count, dtype=np.int64),
        )
    )
    block = (scale >= 1000).astype(np.int8)
    assigned = block.astype(np.int64) * center_count + center
    seeds = np.zeros((2 * center_count, 3), dtype=np.float64)
    center_time = np.zeros((len(scale), 32), dtype=np.float32)
    return {
        "seeds_xyz": seeds,
        "valid_assigned_row_index": assigned,
        "valid_center_seed_index": center,
        "valid_scale_block_index": block,
        "valid_scale_id": scale,
        "center_sample_time": center_time,
    }


def _write_poisoned_parent(path: Path, arrays: dict[str, np.ndarray]):
    count = len(arrays["valid_scale_id"])
    poison = {
        "valid_labels": np.asarray([object() for _ in range(count)], dtype=object),
        "reference_labels_all": np.asarray([object()], dtype=object),
        "ivd_values_all": np.asarray([object()], dtype=object),
        "ivd_volume": np.asarray([object()], dtype=object),
        "metadata_json": np.asarray({"forbidden": True}, dtype=object),
    }
    with path.open("wb") as destination:
        np.savez_compressed(destination, **arrays, **poison)
    hashes = {
        name: canonical_array_sha256(arrays[name])
        for name in PARENT_PROJECTION_MEMBER_NAMES
    }
    return sha256_file(path), hashes


def _projection(scale_ids: np.ndarray) -> ParentKinematicProjection:
    return ParentKinematicProjection(**_parent_arrays(scale_ids))


def _provenance_bindings() -> dict[str, object]:
    values: dict[str, object] = {
        name: f"synthetic-{index}"
        for index, name in enumerate(SIDECAR_PROVENANCE_BINDING_NAMES)
    }
    values["dataset_family_source_identity"] = {
        "dataset": "synthetic",
        "physical_family": "synthetic_family",
        "source_ordinal": 0,
        "source_index": 0,
    }
    return values


def test_narrow_parent_loader_authenticates_only_six_members_and_ignores_poison():
    arrays = _parent_arrays(np.asarray([0, 1000], dtype=np.int32))
    arrays["seeds_xyz"][1] = np.asarray([0.25, -0.125, 0.5])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "parent.npz"
        file_hash, hashes = _write_poisoned_parent(path, arrays)
        projection = load_parent_kinematic_projection(
            path,
            expected_size_bytes=path.stat().st_size,
            expected_file_sha256=file_hash,
            expected_array_sha256=hashes,
        )
        assert projection.opened_member_names == PARENT_PROJECTION_MEMBER_NAMES
        assert FORBIDDEN_PARENT_MEMBER_NAMES.isdisjoint(projection.opened_member_names)
        assert not hasattr(projection, "valid_labels")
        np.testing.assert_array_equal(projection.seeds_xyz[1], [0.25, -0.125, 0.5])
        for name in PARENT_PROJECTION_MEMBER_NAMES:
            assert not np.asarray(getattr(projection, name)).flags.writeable

        bad_hashes = dict(hashes)
        bad_hashes["valid_scale_id"] = "0" * 64
        _expect_value_error(
            load_parent_kinematic_projection,
            path,
            expected_size_bytes=path.stat().st_size,
            expected_file_sha256=file_hash,
            expected_array_sha256=bad_hashes,
        )
        _expect_value_error(
            load_parent_kinematic_projection,
            path,
            expected_size_bytes=path.stat().st_size,
            expected_file_sha256="0" * 64,
            expected_array_sha256=hashes,
        )
        _expect_value_error(
            load_parent_kinematic_projection,
            path,
            expected_size_bytes=path.stat().st_size + 1,
            expected_file_sha256=file_hash,
            expected_array_sha256=hashes,
        )


def test_parent_loader_single_snapshot_rejects_final_path_inode_replacement():
    arrays = _parent_arrays(np.asarray([0, 1000], dtype=np.int32))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "parent.npz"
        file_hash, hashes = _write_poisoned_parent(path, arrays)
        real_from_stat = sidecar_module._FileIdentity.from_stat
        calls = []

        def changed_final_path_identity(value):
            identity = real_from_stat(value)
            calls.append(identity)
            if len(calls) == 4:
                return replace(identity, inode=identity.inode + 1)
            return identity

        with patch.object(
            sidecar_module._FileIdentity,
            "from_stat",
            side_effect=changed_final_path_identity,
        ):
            _expect_value_error(
                load_parent_kinematic_projection,
                path,
                expected_size_bytes=path.stat().st_size,
                expected_file_sha256=file_hash,
                expected_array_sha256=hashes,
            )
        assert len(calls) == 4

    source = inspect.getsource(load_parent_kinematic_projection)
    assert "_read_authenticated_file_snapshot" in source
    assert "np.load(io.BytesIO(snapshot.content)" in source
    assert "np.load(source" not in source
    assert "sha256_file(source)" not in source


def test_parent_projection_rejects_dtype_shape_time_and_identity_drift():
    valid = _parent_arrays(np.asarray([0, 1000], dtype=np.int32))
    mutations = []

    bad = {name: value.copy() for name, value in valid.items()}
    bad["seeds_xyz"] = bad["seeds_xyz"].astype(np.float32)
    mutations.append(bad)
    bad = {name: value.copy() for name, value in valid.items()}
    bad["valid_assigned_row_index"][1] = bad["valid_assigned_row_index"][0]
    mutations.append(bad)
    bad = {name: value.copy() for name, value in valid.items()}
    bad["valid_center_seed_index"][1] = 1
    mutations.append(bad)
    bad = {name: value.copy() for name, value in valid.items()}
    bad["valid_scale_block_index"][1] = 0
    mutations.append(bad)
    bad = {name: value.copy() for name, value in valid.items()}
    bad["center_sample_time"][0, 0] = 1.1e-7
    mutations.append(bad)
    bad = {name: value.copy() for name, value in valid.items()}
    bad["center_sample_time"] = bad["center_sample_time"][:, :-1]
    mutations.append(bad)

    for mutation in mutations:
        _expect_value_error(ParentKinematicProjection, **mutation)


def test_sampler_is_bitwise_equal_to_production_interpolator_and_uses_frame_zero():
    window = _nonlinear_window()
    centers = np.asarray([[0.0, 0.0, 0.0], [0.25, -0.125, 0.0]], dtype=np.float64)
    dx = np.asarray([0.125, 0.0625], dtype=np.float64)
    sampled = sample_seed_time_velocity_xyz(window, centers, dx)
    vector_field = UnsteadyVectorField3D.from_window(window)
    positions = centers[:, None, :] + dx[:, None, None] * _OFFSETS[None, :, :]
    expected = np.empty((2, 7, 3), dtype=np.float32)
    for row in range(2):
        for line in range(7):
            expected[row, line] = np.asarray(
                _interp4_quadrilinear_scalar(
                    vector_field.field,
                    vector_field.domain_min,
                    vector_field.grid_interval,
                    vector_field.xdim,
                    vector_field.ydim,
                    vector_field.zdim,
                    vector_field.tmin,
                    vector_field.time_interval,
                    vector_field.time_steps,
                    *positions[row, line],
                    vector_field.tmin,
                ),
                dtype=np.float32,
            )
    np.testing.assert_array_equal(sampled, expected)
    assert sampled.dtype == np.float32
    assert not sampled.flags.writeable

    changed_second_frame = window.velocity.copy()
    changed_second_frame[1] = -9876.0
    changed_window = replace(window, velocity=changed_second_frame)
    np.testing.assert_array_equal(
        sampled, sample_seed_time_velocity_xyz(changed_window, centers, dx)
    )


def test_sampler_prechecks_domain_and_rejects_bad_portable_or_dx_contracts():
    window = _nonlinear_window()
    _expect_value_error(
        sample_seed_time_velocity_xyz,
        window,
        np.asarray([[0.9, 0.0, 0.0]], dtype=np.float64),
        np.asarray([0.2], dtype=np.float64),
    )
    valid_center = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    for malformed_dx in (
        np.asarray([0.1], dtype=np.float32),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([-0.1], dtype=np.float64),
        np.asarray([np.nan], dtype=np.float64),
        np.asarray([[0.1]], dtype=np.float64),
    ):
        _expect_value_error(
            sample_seed_time_velocity_xyz, window, valid_center, malformed_dx
        )

    _expect_value_error(
        sample_seed_time_velocity_xyz,
        replace(window, velocity=window.velocity.astype(np.float64)),
        valid_center,
        np.asarray([0.1], dtype=np.float64),
    )
    bad_velocity = window.velocity.copy()
    bad_velocity[0, 0, 0, 0, 0] = np.nan
    _expect_value_error(
        sample_seed_time_velocity_xyz,
        replace(window, velocity=bad_velocity),
        valid_center,
        np.asarray([0.1], dtype=np.float64),
    )
    bad_x = window.coordinates_xyz[0].copy()
    bad_x[2] += 0.05
    _expect_value_error(
        sample_seed_time_velocity_xyz,
        replace(window, coordinates_xyz=(bad_x, *window.coordinates_xyz[1:])),
        valid_center,
        np.asarray([0.1], dtype=np.float64),
    )


def test_physical_dx_uses_parent_float32_spacing_and_explicit_2000_scale_table():
    x = np.arange(11, dtype=np.float64) * 0.1
    y = np.arange(11, dtype=np.float64) * 0.2
    z = np.arange(11, dtype=np.float64) * 0.3
    velocity = np.zeros((2, 11, 11, 11, 3), dtype=np.float32)
    window = FlowWindow3D(
        velocity=velocity,
        coordinates_xyz=(x, y, z),
        time=np.asarray([1.0, 1.5], dtype=np.float64),
        source_path="synthetic-spacing.nc",
        source_start_index=0,
        spatial_strides={"x": 1, "y": 1, "z": 1},
        components=("u", "v", "w"),
        coordinate_sources={"x": "synthetic", "y": "synthetic", "z": "synthetic", "t": "synthetic"},
    )
    physical = physical_dx_by_scale_for_window(window)
    expected_spacing = float(np.float32(0.1))
    np.testing.assert_array_equal(
        physical, FROZEN_DX_GRID_SCALE_BY_ID * expected_spacing
    )
    assert physical.shape == (2000,)
    assert len(np.unique(FROZEN_DX_GRID_SCALE_BY_ID[:1000])) == 10
    assert len(np.unique(FROZEN_DX_GRID_SCALE_BY_ID[1000:])) == 10
    assert not physical.flags.writeable


def test_full_2000_scale_payload_uses_assigned_seed_and_matches_affine_oracle():
    matrix = np.asarray(
        [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 1.0, 1.25]],
        dtype=np.float64,
    )
    translation = np.asarray([0.25, -0.5, 1.0], dtype=np.float64)
    window = _affine_window(matrix, translation)
    parent_arrays = _parent_arrays(np.arange(2000, dtype=np.int32))
    # Distinguish assigned-row lookup from the repeated center identity.
    parent_arrays["seeds_xyz"][
        parent_arrays["valid_assigned_row_index"][1000:]
    ] = np.asarray([0.125, -0.125, 0.0])
    parent = ParentKinematicProjection(**parent_arrays)
    payload = build_seed_time_kinematic_sidecar_payload(parent, window)
    assert payload.seed_velocity_xyz.shape == (2000, 7, 3)
    assert payload.seed_kinematic4.shape == (2000, 4)
    assert payload.physical_dx_by_scale.shape == (2000,)
    np.testing.assert_allclose(
        payload.seed_velocity_xyz[1000, 0],
        matrix @ np.asarray([0.125, -0.125, 0.0]) + translation,
        rtol=2e-6,
        atol=2e-6,
    )
    gradient = compute_seed_time_velocity_gradient(
        payload.seed_velocity_xyz,
        payload.physical_dx_by_scale[payload.valid_scale_id],
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )
    np.testing.assert_allclose(
        gradient,
        np.broadcast_to(matrix, gradient.shape),
        rtol=2e-5,
        atol=2e-5,
    )
    validate_sidecar_identity_join(parent, payload)
    for name in SIDECAR_ARRAY_NAMES:
        assert not np.asarray(getattr(payload, name)).flags.writeable


def test_identity_join_rejects_mismatch_and_payload_rejects_feature_tamper():
    parent = _projection(np.asarray([0, 1000], dtype=np.int32))
    payload = build_seed_time_kinematic_sidecar_payload(parent, _affine_window())
    wrong_center = payload.valid_center_seed_index.copy()
    wrong_center[1] += 1
    mismatched = SeedTimeKinematicSidecarPayload(
        valid_assigned_row_index=payload.valid_assigned_row_index,
        valid_center_seed_index=wrong_center,
        valid_scale_block_index=payload.valid_scale_block_index,
        valid_scale_id=payload.valid_scale_id,
        seed_velocity_xyz=payload.seed_velocity_xyz,
        seed_kinematic4=payload.seed_kinematic4,
        physical_dx_by_scale=payload.physical_dx_by_scale,
    )
    _expect_value_error(validate_sidecar_identity_join, parent, mismatched)

    tampered_feature = payload.seed_kinematic4.copy()
    tampered_feature[0, 0] += 0.125
    _expect_value_error(
        SeedTimeKinematicSidecarPayload,
        valid_assigned_row_index=payload.valid_assigned_row_index,
        valid_center_seed_index=payload.valid_center_seed_index,
        valid_scale_block_index=payload.valid_scale_block_index,
        valid_scale_id=payload.valid_scale_id,
        seed_velocity_xyz=payload.seed_velocity_xyz,
        seed_kinematic4=tampered_feature,
        physical_dx_by_scale=payload.physical_dx_by_scale,
    )


def test_sidecar_npz_round_trip_authenticates_exact_members_dtypes_shapes_and_hashes():
    parent = _projection(np.asarray([0, 1000], dtype=np.int32))
    payload = build_seed_time_kinematic_sidecar_payload(parent, _affine_window())
    bindings = _provenance_bindings()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sidecar.npz"
        written = write_seed_time_kinematic_sidecar(
            path, payload, provenance_bindings=bindings
        )
        loaded = load_seed_time_kinematic_sidecar(
            path,
            expected_file_sha256=written.file_sha256,
            expected_provenance_bindings=bindings,
            expected_parent=parent,
        )
        with np.load(path, allow_pickle=False) as archive:
            assert tuple(archive.files) == SIDECAR_ARCHIVE_MEMBER_NAMES
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            for name in SIDECAR_ARRAY_NAMES:
                values = np.asarray(archive[name])
                assert metadata["array_sha256"][name] == canonical_array_sha256(values)
                assert metadata["array_contract"][name] == {
                    "dtype": values.dtype.str,
                    "shape": list(values.shape),
                }
        for name in SIDECAR_ARRAY_NAMES:
            np.testing.assert_array_equal(
                getattr(loaded.payload, name), getattr(payload, name)
            )
            values = np.asarray(getattr(loaded.payload, name))
            assert not values.flags.writeable
            try:
                values.setflags(write=True)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"authenticated sidecar array became writeable: {name}"
                )
        try:
            loaded.metadata["schema"] = "tampered"
        except TypeError:
            pass
        else:
            raise AssertionError("authenticated sidecar metadata became writeable")
        try:
            loaded.metadata["array_sha256"][SIDECAR_ARRAY_NAMES[0]] = "0" * 64
        except TypeError:
            pass
        else:
            raise AssertionError("nested authenticated metadata became writeable")
        try:
            write_seed_time_kinematic_sidecar(
                path, payload, provenance_bindings=bindings
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("sidecar overwrite was accepted")


def test_sidecar_publish_is_atomic_no_replace_under_race_and_fsyncs_parent():
    parent = _projection(np.asarray([0, 1000], dtype=np.int32))
    payload = build_seed_time_kinematic_sidecar_payload(parent, _affine_window())
    bindings = _provenance_bindings()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        successful = root / "successful.npz"
        fsync_calls: list[Path] = []
        real_fsync_parent = sidecar_module._fsync_parent_directory

        def tracking_fsync_parent(parent_path: Path) -> None:
            fsync_calls.append(Path(parent_path))
            real_fsync_parent(Path(parent_path))

        with patch.object(
            sidecar_module,
            "_fsync_parent_directory",
            new=tracking_fsync_parent,
        ):
            write_seed_time_kinematic_sidecar(
                successful,
                payload,
                provenance_bindings=bindings,
            )
        assert fsync_calls == [root, root]

        raced = root / "raced.npz"
        competitor_content = b"competitor-final-must-survive"
        real_link = sidecar_module.os.link

        def competing_link(
            source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            *,
            follow_symlinks: bool = True,
        ) -> None:
            Path(destination).write_bytes(competitor_content)
            real_link(source, destination, follow_symlinks=follow_symlinks)

        try:
            with patch.object(sidecar_module.os, "link", new=competing_link):
                write_seed_time_kinematic_sidecar(
                    raced,
                    payload,
                    provenance_bindings=bindings,
                )
        except FileExistsError:
            pass
        else:
            raise AssertionError("racing sidecar writer overwrote an existing final path")
        assert raced.read_bytes() == competitor_content
        assert list(root.glob(f".{raced.name}.*.partial")) == []


def test_parent_directory_fsync_uses_and_closes_posix_directory_descriptor():
    parent = Path("synthetic-parent")
    events: list[tuple[object, ...]] = []

    def fake_open(path, flags):
        events.append(("open", path, flags))
        return 73

    def fake_fsync(descriptor):
        events.append(("fsync", descriptor))

    def fake_close(descriptor):
        events.append(("close", descriptor))

    directory_flag = 0x10000
    with (
        patch.object(sidecar_module.os, "name", "posix"),
        patch.object(sidecar_module.os, "O_DIRECTORY", directory_flag, create=True),
        patch.object(sidecar_module.os, "open", new=fake_open),
        patch.object(sidecar_module.os, "fsync", new=fake_fsync),
        patch.object(sidecar_module.os, "close", new=fake_close),
    ):
        sidecar_module._fsync_parent_directory(parent)
    assert events == [
        ("open", os.fspath(parent), os.O_RDONLY | directory_flag),
        ("fsync", 73),
        ("close", 73),
    ]


def test_sidecar_loader_rejects_same_size_same_mtime_replacement_during_read():
    parent = _projection(np.asarray([0, 1000], dtype=np.int32))
    payload = build_seed_time_kinematic_sidecar_payload(parent, _affine_window())
    bindings = _provenance_bindings()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "sidecar.npz"
        written = write_seed_time_kinematic_sidecar(
            path,
            payload,
            provenance_bindings=bindings,
        )
        original_stat = path.stat()
        replacement_content = bytearray(path.read_bytes())
        replacement_content[len(replacement_content) // 2] ^= 0x01
        replacement = root / "replacement.npz"
        replacement.write_bytes(replacement_content)
        os.utime(
            replacement,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        assert replacement.stat().st_size == original_stat.st_size
        assert replacement.stat().st_mtime_ns == original_stat.st_mtime_ns

        real_numpy_load = sidecar_module.np.load
        replacement_performed = False

        def replacing_numpy_load(*args, **kwargs):
            nonlocal replacement_performed
            if not replacement_performed:
                os.replace(replacement, path)
                replacement_performed = True
            return real_numpy_load(*args, **kwargs)

        with patch.object(sidecar_module.np, "load", new=replacing_numpy_load):
            _expect_value_error(
                load_seed_time_kinematic_sidecar,
                path,
                expected_file_sha256=written.file_sha256,
                expected_provenance_bindings=bindings,
                expected_parent=parent,
            )
        assert replacement_performed
        assert path.stat().st_size == original_stat.st_size
        assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
        assert sha256_file(path) != written.file_sha256


def test_sidecar_metadata_rejects_label_derived_positive_and_negative_counts():
    parent = _projection(np.asarray([0, 1000], dtype=np.int32))
    payload = build_seed_time_kinematic_sidecar_payload(parent, _affine_window())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, forbidden_key in enumerate(
            ("positive_count", "negative_count", "positive_fraction", "negative_sample_count")
        ):
            forbidden_bindings = _provenance_bindings()
            forbidden_bindings["dataset_family_source_identity"][forbidden_key] = 1
            _expect_value_error(
                write_seed_time_kinematic_sidecar,
                root / f"forbidden_{index}.npz",
                payload,
                provenance_bindings=forbidden_bindings,
            )

        original = root / "original.npz"
        write_seed_time_kinematic_sidecar(
            original,
            payload,
            provenance_bindings=_provenance_bindings(),
        )
        with np.load(original, allow_pickle=False) as archive:
            stored = {name: np.array(archive[name], copy=True) for name in archive.files}
        metadata = json.loads(str(stored["metadata_json"].item()))
        metadata["provenance_bindings"]["dataset_family_source_identity"][
            "positive_count"
        ] = 1
        stored["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        forged_nested = root / "forged_nested.npz"
        np.savez_compressed(forged_nested, **stored)
        _expect_value_error(load_seed_time_kinematic_sidecar, forged_nested)

        metadata.pop("provenance_bindings")
        metadata["provenance_bindings"] = _provenance_bindings()
        metadata["negative_count"] = 1
        stored["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        forged_top_level = root / "forged_top_level.npz"
        np.savez_compressed(forged_top_level, **stored)
        _expect_value_error(load_seed_time_kinematic_sidecar, forged_top_level)


def test_sidecar_archive_tamper_extra_member_and_forbidden_metadata_fail_closed():
    parent = _projection(np.asarray([0, 1000], dtype=np.int32))
    payload = build_seed_time_kinematic_sidecar_payload(parent, _affine_window())
    bindings = _provenance_bindings()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original = root / "original.npz"
        write_seed_time_kinematic_sidecar(
            original, payload, provenance_bindings=bindings
        )
        with np.load(original, allow_pickle=False) as archive:
            stored = {name: np.array(archive[name], copy=True) for name in archive.files}

        tampered = dict(stored)
        tampered["seed_velocity_xyz"] = tampered["seed_velocity_xyz"].copy()
        tampered["seed_velocity_xyz"][0, 0, 0] += 1.0
        tampered_path = root / "tampered.npz"
        np.savez_compressed(tampered_path, **tampered)
        _expect_value_error(load_seed_time_kinematic_sidecar, tampered_path)

        extra = dict(stored)
        extra["unexpected"] = np.asarray(1, dtype=np.int64)
        extra_path = root / "extra.npz"
        np.savez_compressed(extra_path, **extra)
        _expect_value_error(load_seed_time_kinematic_sidecar, extra_path)

        missing = dict(stored)
        del missing["seed_kinematic4"]
        missing_path = root / "missing.npz"
        np.savez_compressed(missing_path, **missing)
        _expect_value_error(load_seed_time_kinematic_sidecar, missing_path)

        dtype_drift = dict(stored)
        dtype_drift["valid_scale_id"] = dtype_drift["valid_scale_id"].astype(np.int64)
        dtype_path = root / "dtype.npz"
        np.savez_compressed(dtype_path, **dtype_drift)
        _expect_value_error(load_seed_time_kinematic_sidecar, dtype_path)

        wrong_bindings = dict(bindings)
        wrong_bindings["dataset_family_source_identity"] = {
            "dataset": "other_synthetic",
            "physical_family": "synthetic_family",
            "source_ordinal": 0,
            "source_index": 0,
        }
        _expect_value_error(
            load_seed_time_kinematic_sidecar,
            original,
            expected_provenance_bindings=wrong_bindings,
        )

        forbidden_bindings = dict(bindings)
        forbidden_bindings[
            "parent_cache_path_size_file_sha256_and_allowed_array_hashes"
        ] = {"valid_labels": "forbidden"}
        _expect_value_error(
            write_seed_time_kinematic_sidecar,
            root / "forbidden.npz",
            payload,
            provenance_bindings=forbidden_bindings,
        )
