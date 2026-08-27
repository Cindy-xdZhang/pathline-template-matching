"""Training-free Fourier descriptors for 3D pathline-cross primitives.

The primitive layout is ``[N, K, L, C]`` where line 0 is the center and
lines 1..K-1 are spatial neighbours.  The intended 3D cross has K=7 in the
order center, x+, x-, y+, y-, z+, z-.  C is either (x,y,z) or (x,y,z,t).
"""

from __future__ import annotations

import torch


def time_local_gram_dft_features_3d(
    pathlines: torch.Tensor,
    num_freq: int = 6,
    subtract_initial: bool = True,
    normalize_initial_scale: bool = True,
    eps: float = 1e-8,
    return_numpy: bool = True,
):
    """Encode frame-indifferent neighbour deformation before temporal DFT.

    For every time sample, the six neighbour offsets relative to the centre
    line form a ``6 x 3`` matrix ``D(t)``.  Its Gram matrix
    ``G(t) = D(t) D(t)^T`` is unchanged by an arbitrary time-dependent rigid
    observer transformation because translation cancels and
    ``D(t) Q(t)^T Q(t) D(t)^T = G(t)``.  The upper-triangular 21 scalar time
    series are then encoded by their real Fourier coefficients.

    The output contains ``num_freq`` real and ``num_freq - 1`` imaginary
    coefficients for every Gram entry; the identically-zero imaginary DC
    coefficient is omitted.  With six frequencies this gives 231 features.
    Neighbour order is semantic and must remain
    ``x+, x-, y+, y-, z+, z-``.
    """
    pathlines = torch.as_tensor(pathlines)
    if pathlines.ndim != 4 or pathlines.shape[1] != 7 or pathlines.shape[-1] < 3:
        raise ValueError(
            "pathlines must be [N,7,L,C>=3] with centre then x+/x-/y+/y-/z+/z-"
        )
    if pathlines.shape[2] < 2:
        raise ValueError("pathlines must contain at least two time samples")
    independent_bins = pathlines.shape[2] // 2 + 1
    if not 1 <= int(num_freq) <= independent_bins:
        raise ValueError(
            f"num_freq={num_freq} must be in [1,{independent_bins}] "
            f"for L={pathlines.shape[2]}"
        )

    xyz = pathlines[..., :3]
    if not xyz.is_floating_point():
        xyz = xyz.float()
    relative = xyz[:, 1:] - xyz[:, :1]
    relative_t = relative.permute(0, 2, 1, 3)
    gram = torch.einsum("ntic,ntjc->ntij", relative_t, relative_t)
    upper = torch.triu_indices(6, 6, device=gram.device)
    series = gram[:, :, upper[0], upper[1]]

    if normalize_initial_scale:
        initial = gram[:, 0]
        scale = initial.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        series = series / scale.clamp_min(float(eps))[:, None, None]
    if subtract_initial:
        series = series - series[:, :1]

    spectrum = torch.fft.rfft(series, dim=1)[:, :int(num_freq)]
    real = spectrum.real.transpose(1, 2).flatten(1)
    imag = spectrum.imag[:, 1:].transpose(1, 2).flatten(1)
    features = torch.cat((real, imag), dim=1)
    return features.detach().cpu().numpy() if return_numpy else features


def pathline_velocity_gradient_scalar_sequences_3d(
    pathlines: torch.Tensor,
    eps: float = 1e-6,
    *,
    sample_times: torch.Tensor | None = None,
    log_compress: bool = False,
):
    """Estimate four local kinematic scalar sequences from a pathline cross.

    Opposite neighbour pairs estimate the local flow-map differential ``D``.
    ``L = D_dot D^+`` approximates the velocity gradient.  A spatially
    uniform observer angular velocity adds the same skew term to every seed;
    subtracting the per-timeslice mean vorticity therefore yields an
    objective vorticity-deviation magnitude in the continuous limit.

    The returned channel order is vorticity-deviation norm, strain Frobenius
    norm, absolute divergence, and a signed Q-like balance between rotation
    deviation and strain.  Time derivatives use the sampled pathline index by
    default for backward compatibility.  Variable-scale Task5 callers should
    provide exact ``sample_times``.  ``log_compress`` applies a signed
    ``log1p`` to prevent rare nearly singular neighbour crosses from
    dominating train statistics.
    """
    pathlines = torch.as_tensor(pathlines)
    if pathlines.ndim != 4 or pathlines.shape[1] != 7 or pathlines.shape[-1] < 3:
        raise ValueError(
            "pathlines must be [N,7,L,C>=3] with centre then x+/x-/y+/y-/z+/z-"
        )
    length = int(pathlines.shape[2])
    if length < 3:
        raise ValueError("velocity-gradient features require at least three samples")
    xyz = pathlines[..., :3]
    if not xyz.is_floating_point():
        xyz = xyz.float()

    # Columns are the three opposite-pair separation vectors.
    pair_vectors = torch.stack(
        (xyz[:, 1] - xyz[:, 2], xyz[:, 3] - xyz[:, 4],
         xyz[:, 5] - xyz[:, 6]),
        dim=-1,
    )
    derivative = torch.empty_like(pair_vectors)
    if sample_times is None:
        # Backward-compatible fixed-scale behaviour: sampled index is time.
        derivative[:, 0] = pair_vectors[:, 1] - pair_vectors[:, 0]
        derivative[:, -1] = pair_vectors[:, -1] - pair_vectors[:, -2]
        derivative[:, 1:-1] = 0.5 * (
            pair_vectors[:, 2:] - pair_vectors[:, :-2]
        )
    else:
        times = torch.as_tensor(
            sample_times, device=pair_vectors.device, dtype=pair_vectors.dtype
        )
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(len(pair_vectors), -1)
        if times.shape != pair_vectors.shape[:2]:
            raise ValueError(
                "sample_times must have shape [L] or [N,L], got "
                f"{tuple(times.shape)} for pathlines {tuple(pathlines.shape)}"
            )
        intervals = times[:, 1:] - times[:, :-1]
        if not torch.isfinite(times).all() or torch.any(intervals <= 0):
            raise ValueError("sample_times must be finite and strictly increasing")
        derivative[:, 0] = (
            pair_vectors[:, 1] - pair_vectors[:, 0]
        ) / intervals[:, 0, None, None]
        derivative[:, -1] = (
            pair_vectors[:, -1] - pair_vectors[:, -2]
        ) / intervals[:, -1, None, None]
        derivative[:, 1:-1] = (
            pair_vectors[:, 2:] - pair_vectors[:, :-2]
        ) / (times[:, 2:] - times[:, :-2])[:, :, None, None]
    inverse = torch.linalg.pinv(pair_vectors, rtol=float(eps))
    gradient = derivative @ inverse

    vorticity = torch.stack(
        (gradient[..., 2, 1] - gradient[..., 1, 2],
         gradient[..., 0, 2] - gradient[..., 2, 0],
         gradient[..., 1, 0] - gradient[..., 0, 1]),
        dim=-1,
    )
    vorticity_deviation = vorticity - vorticity.mean(dim=0, keepdim=True)
    ivd_like = torch.linalg.vector_norm(vorticity_deviation, dim=-1)
    strain = 0.5 * (gradient + gradient.transpose(-1, -2))
    strain_norm = torch.linalg.matrix_norm(strain, ord="fro")
    divergence = gradient.diagonal(dim1=-2, dim2=-1).sum(dim=-1).abs()
    q_like = 0.25 * ivd_like.square() - 0.5 * strain_norm.square()
    series = torch.stack((ivd_like, strain_norm, divergence, q_like), dim=-1)
    if log_compress:
        unsigned = torch.log1p(series[..., :3].clamp_min(0.0))
        signed_q = torch.sign(series[..., 3:]) * torch.log1p(
            series[..., 3:].abs()
        )
        series = torch.cat((unsigned, signed_q), dim=-1)

    if not torch.isfinite(series).all():
        raise ValueError("non-finite pathline velocity-gradient sequences")
    return series


def pathline_velocity_gradient_dft_features_3d(
    pathlines: torch.Tensor,
    num_freq: int = 6,
    eps: float = 1e-6,
    return_numpy: bool = True,
    *,
    sample_times: torch.Tensor | None = None,
    log_compress: bool = False,
):
    """Fourier encode local kinematic scalar sequences from a pathline cross."""
    series = pathline_velocity_gradient_scalar_sequences_3d(
        pathlines,
        eps=eps,
        sample_times=sample_times,
        log_compress=log_compress,
    )
    independent_bins = series.shape[1] // 2 + 1
    if not 1 <= int(num_freq) <= independent_bins:
        raise ValueError(
            f"num_freq={num_freq} must be in [1,{independent_bins}] "
            f"for L={series.shape[1]}"
        )

    spectrum = torch.fft.rfft(series, dim=1)[:, :int(num_freq)]
    real = spectrum.real.transpose(1, 2).flatten(1)
    imag = spectrum.imag[:, 1:].transpose(1, 2).flatten(1)
    features = torch.cat((real, imag), dim=1)
    if not torch.isfinite(features).all():
        raise ValueError("non-finite pathline velocity-gradient features")
    return features.detach().cpu().numpy() if return_numpy else features


def pathline_anchored_kinematic_dft_features_3d(
    pathlines: torch.Tensor,
    num_freq: int = 6,
    window: int | None = None,
    channels: tuple[int, ...] = (0, 1, 2, 3),
    eps: float = 1e-6,
    return_numpy: bool = True,
    *,
    sample_times: torch.Tensor | None = None,
    log_compress: bool = False,
):
    """Combine early-window Fourier coefficients with time-domain anchors.

    Task3 labels describe the seed time, while a low-frequency temporal DFT
    summarizes the complete integration window.  This parameter-free block
    keeps the DFT and appends seven anchors per selected scalar sequence:
    first value, early-quarter mean, full-window mean, standard deviation,
    maximum, minimum, and last value.  It therefore preserves seed-local
    evidence without discarding later pathline deformation.

    Channel indices follow
    :func:`pathline_velocity_gradient_scalar_sequences_3d`.
    """
    series = pathline_velocity_gradient_scalar_sequences_3d(
        pathlines,
        eps=eps,
        sample_times=sample_times,
        log_compress=log_compress,
    )
    selected = tuple(int(value) for value in channels)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("channels must contain unique kinematic channel indices")
    if min(selected) < 0 or max(selected) >= series.shape[-1]:
        raise ValueError(f"invalid kinematic channels: {selected}")
    series = series[..., list(selected)]
    if window is not None:
        window = int(window)
        if not 3 <= window <= series.shape[1]:
            raise ValueError(
                f"window={window} must be in [3,{series.shape[1]}]"
            )
        series = series[:, :window]

    independent_bins = series.shape[1] // 2 + 1
    if not 1 <= int(num_freq) <= independent_bins:
        raise ValueError(
            f"num_freq={num_freq} must be in [1,{independent_bins}] "
            f"for window={series.shape[1]}"
        )
    spectrum = torch.fft.rfft(series, dim=1)[:, :int(num_freq)]
    dft = torch.cat(
        (
            spectrum.real.transpose(1, 2).flatten(1),
            spectrum.imag[:, 1:].transpose(1, 2).flatten(1),
        ),
        dim=1,
    )
    early_count = max(2, int(series.shape[1]) // 4)
    anchors = torch.stack(
        (
            series[:, 0],
            series[:, :early_count].mean(dim=1),
            series.mean(dim=1),
            series.std(dim=1, unbiased=False),
            series.amax(dim=1),
            series.amin(dim=1),
            series[:, -1],
        ),
        dim=-1,
    ).flatten(1)
    features = torch.cat((dft, anchors), dim=1)
    if not torch.isfinite(features).all():
        raise ValueError("non-finite anchored kinematic Fourier features")
    return features.detach().cpu().numpy() if return_numpy else features


def fmt_feature_indices_3d(name="all", num_freq=6, line_count=7,
                           include_chirality=True):
    """Return semantic FMT feature indices for label-free block ablations."""
    gram_width = int(num_freq) * 3
    block_width = gram_width + (int(num_freq) - 1 if include_chirality else 0)
    slots = torch.arange(int(line_count) * block_width).reshape(line_count, block_width)
    semantic = {
        "real": torch.arange(0, gram_width, 3),
        "imag": torch.arange(1, gram_width, 3),
        "cosine": torch.arange(2, gram_width, 3),
        "chirality": torch.arange(gram_width, block_width),
    }
    if name == "all":
        return slots.flatten().numpy()
    parts = str(name).split("_")
    if parts[-1] in {"center", "neighbor", "all"}:
        scope = parts.pop()
    else:
        scope = "all"
    if not parts or any(part not in semantic for part in parts):
        raise ValueError(f"unknown FMT feature subset: {name!r}")
    lines = {"center": [0], "neighbor": range(1, line_count),
             "all": range(line_count)}[scope]
    local = torch.cat([semantic[part] for part in parts])
    return torch.cat([slots[line, local] for line in lines]).numpy()


def dft_rotation_invariants_3d(
    seq: torch.Tensor,
    num_freq: int,
    mode: str = "gram",
    include_chirality: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encode ``[B,T,3]`` real vector sequences with SO(3)-invariant features.

    A constant rotation R maps each complex Fourier vector F[k] to R F[k].
    Therefore its norm and the Gram invariants of (Re F, Im F) are unchanged.
    Optional triple products preserve handedness under proper rotations while
    changing sign under reflections.
    """
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"seq must be [B,T,3], got {tuple(seq.shape)}")
    if seq.shape[1] < 1:
        raise ValueError("sequence length must be positive")

    independent_bins = seq.shape[1] // 2 + 1
    if not 1 <= int(num_freq) <= independent_bins:
        raise ValueError(
            f"num_freq={num_freq} must be in [1,{independent_bins}] for T={seq.shape[1]}"
        )
    if include_chirality and num_freq < 2:
        raise ValueError("include_chirality=True requires num_freq >= 2")

    work = seq if seq.dtype in (torch.float32, torch.float64) else seq.float()
    spectrum = torch.fft.rfft(work, dim=1)[:, :num_freq, :]
    real, imag = spectrum.real, spectrum.imag

    if mode == "magnitude":
        features = torch.linalg.vector_norm(spectrum, dim=-1)
    elif mode == "gram":
        real_norm = torch.linalg.vector_norm(real, dim=-1)
        imag_norm = torch.linalg.vector_norm(imag, dim=-1)
        cosine = (real * imag).sum(dim=-1) / (real_norm * imag_norm).clamp_min(eps)
        features = torch.stack((real_norm, imag_norm, cosine), dim=-1).flatten(1)
    else:
        raise ValueError(f"mode must be 'magnitude' or 'gram', got {mode!r}")

    if include_chirality:
        # The first value is necessarily zero because Im(F[0])=0.  Keep the
        # fixed-width slot explicitly so feature dimensions remain predictable.
        cross = torch.cross(real[:, :-1], imag[:, :-1], dim=-1)
        triple = (cross * real[:, 1:]).sum(dim=-1)
        denom = (
            torch.linalg.vector_norm(real[:, :-1], dim=-1)
            * torch.linalg.vector_norm(imag[:, :-1], dim=-1)
            * torch.linalg.vector_norm(real[:, 1:], dim=-1)
        ).clamp_min(eps)
        features = torch.cat((features, triple / denom), dim=-1)
    return features


def pathline_dft_features_3d(
    pathlines: torch.Tensor,
    valid_index: torch.Tensor | None = None,
    num_freq: int = 6,
    neighbor_weight: float = 0.5,
    neighbor_scale: float = 100.0,
    neighbor_pool: str = "sort",
    mode: str = "gram",
    include_chirality: bool = True,
    return_numpy: bool = True,
):
    """Encode 3D pathline primitives into one feature vector per seed.

    Temporal differences remove constant translations.  Neighbour trajectories
    are first expressed relative to the center pathline.  ``sort`` pools each
    feature slot across neighbours and is invariant to neighbour relabelling.
    """
    pathlines = torch.as_tensor(pathlines)
    if pathlines.ndim != 4:
        raise ValueError(f"pathlines must be [N,K,L,C], got {tuple(pathlines.shape)}")
    _, line_count, length, channels = pathlines.shape
    if line_count < 2 or length < 2 or channels not in (3, 4):
        raise ValueError(
            f"expected K>=2, L>=2 and C in {{3,4}}, got {tuple(pathlines.shape)}"
        )
    xyz = pathlines[..., :3]
    if not xyz.is_floating_point():
        xyz = xyz.float()
    if valid_index is not None:
        index = torch.as_tensor(valid_index, device=xyz.device)
        xyz = xyz[index]

    center = xyz[:, :1]
    neighbours = xyz[:, 1:]
    center_delta = center[:, :, 1:] - center[:, :, :-1]
    relative = neighbours - center
    neighbour_delta = (relative[:, :, 1:] - relative[:, :, :-1]) * neighbor_scale

    n_valid, neighbour_count, seq_len, _ = neighbour_delta.shape
    center_features = dft_rotation_invariants_3d(
        center_delta[:, 0], num_freq, mode, include_chirality
    )
    block_width = center_features.shape[-1]
    neighbour_features = dft_rotation_invariants_3d(
        neighbour_delta.reshape(n_valid * neighbour_count, seq_len, 3),
        num_freq,
        mode,
        include_chirality,
    ).reshape(n_valid, neighbour_count, block_width)

    if neighbor_pool == "none":
        pooled = neighbour_features.flatten(1)
    elif neighbor_pool == "sort":
        pooled = neighbour_features.sort(dim=1, descending=True).values.flatten(1)
    elif neighbor_pool == "mean":
        pooled = neighbour_features.mean(dim=1)
    elif neighbor_pool == "max":
        pooled = neighbour_features.amax(dim=1)
    else:
        raise ValueError("neighbor_pool must be one of: none, sort, mean, max")

    result = torch.cat((center_features, neighbor_weight * pooled), dim=-1)
    return result.detach().cpu().numpy() if return_numpy else result
