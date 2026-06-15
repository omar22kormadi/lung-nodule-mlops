"""
LUNA16 YOLO Dataset Preprocessing Pipeline v2
==============================================
Production pipeline addressing v1 failure modes:
- Lung-mask constrained slice selection
- Visibility-aware bounding boxes (no blind z±3 propagation)
- Smart negative sampling with nodule exclusion zones
- Correct patient-level splits (no _neg suffix)
- Configurable clinical HU windowing
- Full diagnostics, QA, and v1 vs v2 comparison before export

Author: CRISP-ML(Q)
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage import measure, morphology
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class WindowPreset:
    """CT display window: clip [center - width/2, center + width/2] then scale to uint8."""

    name: str
    center: float
    width: float

    @property
    def hu_min(self) -> float:
        return self.center - self.width / 2.0

    @property
    def hu_max(self) -> float:
        return self.center + self.width / 2.0


@dataclass
class PipelineConfig:
    """All tunable thresholds — no magic numbers in processing logic."""

    # Paths
    luna16_base_h: Path = Path("H:/luna16")
    luna16_base_g: Path = Path("G:/luna16")
    subsets_g: tuple = (0, 1)
    subsets_h: tuple = (2, 3, 4, 5, 6, 7, 8, 9)
    output_dir: Path = field(
        default_factory=lambda: Path(
            r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\luna16_yolo_dataset_v4"
        )
    )
    legacy_dataset_dir: Path = field(
        default_factory=lambda: Path(
            r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\luna16_yolo_dataset"
        )
    )
    legacy_audit_csv: Path = field(
        default_factory=lambda: Path(
            r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\luna16_yolo_dataset\audit_report\slice_metrics.csv"
        )
    )

    # Geometry
    target_spacing: tuple = (1.0, 1.0, 1.0)

    # HU windowing (default: clinical lung window)
    window: WindowPreset = field(
        default_factory=lambda: WindowPreset("lung_default", center=-600.0, width=1500.0)
    )
    generate_window_comparisons: bool = True
    window_comparison_scans: int = 3

    # Lung segmentation (HU-based, validated on chest CT)
    lung_hu_low: float = -1024.0
    lung_hu_high: float = -320.0
    morph_ball_radius_mm: float = 2.0
    min_lung_voxels_3d: int = 50_000

    # Per-slice quality gates
    min_lung_coverage: float = 0.08
    max_lung_coverage: float = 0.72
    min_lung_pixels: int = 6_000
    max_slice_mean_hu: float = 35.0
    max_soft_tissue_fraction: float = 0.52

    # Nodule / bbox
    min_nodule_diameter_mm: float = 3.0
    min_bbox_pixels: float = 6.0
    max_bbox_pixels: float = 120.0
    min_bbox_local_contrast: float = -8.0
    nodule_visibility_margin_mm: float = 0.5

    # Negative sampling
    negative_ratio: float = 1.25
    max_negatives_per_scan: int = 80
    nodule_exclusion_mm: float = 12.0
    min_negative_pool: int = 5

    # Split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Export
    image_compression: int = 6
    use_25d_stacking: bool = True  # Enable 2.5D RGB stacking (R=Z-2, G=Z, B=Z+2)
    slice_gap_25d: int = 2  # Gap between slices for better depth context (Z-gap, Z, Z+gap)
    qa_overlay_samples: int = 24
    log_level: int = logging.INFO

    @classmethod
    def presets(cls) -> dict[str, WindowPreset]:
        return {
            "lung_default": WindowPreset("lung_default", -600.0, 1500.0),
            "lung_narrow": WindowPreset("lung_narrow", -600.0, 1200.0),
            "mediastinal": WindowPreset("mediastinal", 40.0, 400.0),
            "legacy_v1": WindowPreset("legacy_v1", -300.0, 1400.0),
        }


# =============================================================================
# LOGGING
# =============================================================================


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("LUNA16_v2")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# =============================================================================
# SCAN I/O
# =============================================================================


class ScanLoader:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def get_scan_path(self, series_uid: str) -> Optional[Path]:
        for subsets, base in [
            (self.cfg.subsets_g, self.cfg.luna16_base_g),
            (self.cfg.subsets_h, self.cfg.luna16_base_h),
        ]:
            for subset in subsets:
                p = base / f"subset{subset}" / f"{series_uid}.mhd"
                if p.exists():
                    return p
        return None

    def load_hu_volume(self, series_uid: str) -> Optional[tuple[np.ndarray, tuple, tuple, sitk.Image]]:
        path = self.get_scan_path(series_uid)
        if path is None:
            self.logger.warning("Scan not found: %s", series_uid)
            return None
        try:
            img = sitk.ReadImage(str(path), sitk.sitkFloat32)
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            
            # PRESERVE native XY resolution (usually 512x512), only resample Z
            dynamic_target_spacing = (spacing[0], spacing[1], self.cfg.target_spacing[2])
            resampled = self._resample(img, dynamic_target_spacing)
            
            hu = sitk.GetArrayFromImage(resampled).astype(np.float32)
            origin_r = resampled.GetOrigin()
            spacing_r = resampled.GetSpacing()
            return hu, spacing_r, origin_r, resampled
        except Exception as e:
            self.logger.error("Load failed %s: %s", series_uid, e)
            return None

    @staticmethod
    def _resample(image: sitk.Image, target_spacing: tuple) -> sitk.Image:
        sp = image.GetSpacing()
        sz = image.GetSize()
        new_size = [
            int(round(sz[i] * sp[i] / target_spacing[i])) for i in range(3)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024.0)
        return resampler.Execute(image)


# =============================================================================
# COORDINATES
# =============================================================================


class CoordinateTransformer:
    @staticmethod
    def world_to_voxel(
        world: tuple[float, float, float],
        spacing: tuple,
        origin: tuple,
    ) -> tuple[float, float, float]:
        ox, oy, oz = origin
        sx, sy, sz = spacing
        x, y, z = world
        xo = (x - ox) / sx
        yo = (y - oy) / sy
        zo = (z - oz) / sz
        return xo, yo, zo

    @staticmethod
    def clamp_voxel(x: float, y: float, z: float, shape: tuple) -> tuple[int, int, int]:
        zz, yy, xx = shape
        return (
            int(np.clip(z, 0, zz - 1)),
            int(np.clip(y, 0, yy - 1)),
            int(np.clip(x, 0, xx - 1)),
        )


# =============================================================================
# LUNG SEGMENTATION & SLICE QUALITY
# =============================================================================


class LungSegmenter:
    """
    Per-slice 2D lung mask (memory-safe for large volumes).
    HU band [-1024, -320] + 2D morphology; two largest components per slice ≈ lungs.
    Avoids 3D ball ops that allocate full-volume bool arrays (OOM on thick scans).
    """

    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def segment(self, hu_volume: np.ndarray, spacing: tuple) -> np.ndarray:
        cfg = self.cfg
        n_z = hu_volume.shape[0]
        mask = np.zeros(hu_volume.shape, dtype=np.uint8)
        r = max(1, int(round(cfg.morph_ball_radius_mm / spacing[0])))
        selem = morphology.disk(r)

        for z in range(n_z):
            sl = (hu_volume[z] >= cfg.lung_hu_low) & (hu_volume[z] <= cfg.lung_hu_high)
            sl = morphology.binary_closing(sl, selem)
            sl = morphology.binary_opening(sl, selem)
            sl = self._keep_largest_components_2d(sl, n=2)
            mask[z] = sl.astype(np.uint8)

        if int(mask.sum()) < cfg.min_lung_voxels_3d:
            self.logger.debug("Weak lung segmentation: %d voxels", int(mask.sum()))
        return mask

    @staticmethod
    def _keep_largest_components_2d(mask: np.ndarray, n: int = 2) -> np.ndarray:
        labeled = measure.label(mask, connectivity=2)
        if labeled.max() == 0:
            return mask
        regions = measure.regionprops(labeled)
        regions = sorted(regions, key=lambda r: r.area, reverse=True)[:n]
        out = np.zeros(mask.shape, dtype=bool)
        for reg in regions:
            out[labeled == reg.label] = True
        return out


@dataclass
class SliceQuality:
    z: int
    lung_coverage: float
    lung_pixels: int
    mean_hu: float
    soft_tissue_fraction: float
    is_valid: bool
    reject_reason: str = ""


class SliceQualityAnalyzer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def analyze_volume(
        self, hu_volume: np.ndarray, lung_mask: np.ndarray
    ) -> list[SliceQuality]:
        cfg = self.cfg
        results: list[SliceQuality] = []
        n_z = hu_volume.shape[0]
        for z in range(n_z):
            lung_m = lung_mask[z] > 0
            total = lung_m.size
            lung_pixels = int(lung_m.sum())
            lung_cov = lung_pixels / total
            sl = hu_volume[z]
            mean_hu = float(sl.mean())
            soft = ((sl > -100) & (sl < 200) & (~lung_m)).sum() / total

            reason = ""
            valid = True
            if lung_cov < cfg.min_lung_coverage or lung_pixels < cfg.min_lung_pixels:
                valid, reason = False, "low_lung_coverage"
            elif lung_cov > cfg.max_lung_coverage:
                valid, reason = False, "excessive_lung_field"
            elif mean_hu > cfg.max_slice_mean_hu and lung_cov < 0.15:
                valid, reason = False, "bright_non_lung"
            elif soft > cfg.max_soft_tissue_fraction:
                valid, reason = False, "soft_tissue_dominant"

            results.append(
                SliceQuality(
                    z=z,
                    lung_coverage=lung_cov,
                    lung_pixels=lung_pixels,
                    mean_hu=mean_hu,
                    soft_tissue_fraction=float(soft),
                    is_valid=valid,
                    reject_reason=reason,
                )
            )
        return results

    def valid_z_indices(self, qualities: list[SliceQuality]) -> list[int]:
        return [q.z for q in qualities if q.is_valid]


# =============================================================================
# WINDOWING
# =============================================================================


class HUWindowing:
    def __init__(self, preset: WindowPreset):
        self.preset = preset

    def apply(self, hu_slice: np.ndarray) -> np.ndarray:
        lo, hi = self.preset.hu_min, self.preset.hu_max
        clipped = np.clip(hu_slice, lo, hi)
        norm = (clipped - lo) / max(hi - lo, 1e-6)
        return (norm * 255.0).astype(np.uint8)

    def apply_volume(self, hu: np.ndarray) -> np.ndarray:
        lo, hi = self.preset.hu_min, self.preset.hu_max
        clipped = np.clip(hu, lo, hi)
        norm = (clipped - lo) / max(hi - lo, 1e-6)
        return (norm * 255.0).astype(np.uint8)


# =============================================================================
# NODULE SLICES & BBOX
# =============================================================================


@dataclass
class SampleRecord:
    series_uid: str
    slice_index: int
    is_positive: bool
    bbox: Optional[tuple[float, float, float, float]]
    lung_coverage: float
    diameter_mm: float = 0.0
    nodule_id: str = ""
    reject_reason: str = ""
    bbox_contrast: float = 0.0
    split: str = ""


class NoduleSliceExtractor:
    """
    Annotate only slices where the 3D nodule sphere intersects the plane.
    In-plane diameter: 2 * sqrt(R² - dz²) per slice (standard cross-section).
    """

    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.ct = CoordinateTransformer()

    def extract(
        self,
        hu_volume: np.ndarray,
        lung_mask: np.ndarray,
        nodules_df: pd.DataFrame,
        spacing: tuple,
        origin: tuple,
        qualities: list[SliceQuality],
        crop_offset: tuple[int, int] = (0, 0),
    ) -> tuple[list[SampleRecord], set[int]]:
        cfg = self.cfg
        positives: list[SampleRecord] = []
        used_z: set[int] = set()
        windowed = HUWindowing(cfg.window)

        for idx, nodule in nodules_df.iterrows():
            world = (nodule["coordX"], nodule["coordY"], nodule["coordZ"])
            d_mm = float(nodule["diameter_mm"])
            if d_mm < cfg.min_nodule_diameter_mm:
                continue

            vx, vy, vz = self.ct.world_to_voxel(
                world, spacing, origin
            )
            
            # Adjust voxel coordinates for tight lung crop
            vx -= crop_offset[0]
            vy -= crop_offset[1]
            
            zi, yi, xi = self.ct.clamp_voxel(vx, vy, vz, hu_volume.shape)
            radius_mm = d_mm / 2.0 + cfg.nodule_visibility_margin_mm
            radius_z = radius_mm / spacing[2]

            z_lo = max(0, int(np.floor(vz - radius_z)))
            z_hi = min(hu_volume.shape[0], int(np.ceil(vz + radius_z)) + 1)

            nodule_id = f"{nodule['seriesuid']}_{idx}"

            for z in range(z_lo, z_hi):
                dz_mm = abs(z - vz) * spacing[2]
                if dz_mm > radius_mm:
                    continue

                q = qualities[z]
                if not q.is_valid:
                    continue

                if lung_mask[z, int(yi), int(xi)] == 0:
                    continue

                r_plane_mm = float(np.sqrt(max(0.0, radius_mm**2 - dz_mm**2)))
                diam_px = max(
                    2.0 * r_plane_mm / spacing[0],
                    cfg.min_bbox_pixels,
                )
                diam_px = min(diam_px, cfg.max_bbox_pixels)

                bbox = self._yolo_bbox(xi, yi, diam_px, diam_px, hu_volume.shape[1:])
                if bbox is None:
                    continue

                u8 = windowed.apply(hu_volume[z])
                contrast = self._bbox_contrast(u8, bbox)
                if contrast < cfg.min_bbox_local_contrast:
                    self.logger.debug(
                        "Low contrast bbox %s z=%d contrast=%.1f",
                        nodule_id,
                        z,
                        contrast,
                    )
                    continue

                positives.append(
                    SampleRecord(
                        series_uid=nodule["seriesuid"],
                        slice_index=z,
                        is_positive=True,
                        bbox=bbox,
                        lung_coverage=q.lung_coverage,
                        diameter_mm=d_mm,
                        nodule_id=nodule_id,
                        bbox_contrast=contrast,
                    )
                )
                used_z.add(z)

        return positives, used_z

    @staticmethod
    def _yolo_bbox(
        cx: float, cy: float, w_px: float, h_px: float, shape: tuple
    ) -> Optional[tuple[float, float, float, float]]:
        h, w = shape
        x1 = max(0.0, cx - w_px / 2)
        y1 = max(0.0, cy - h_px / 2)
        x2 = min(w, cx + w_px / 2)
        y2 = min(h, cy + h_px / 2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (
            float(np.clip(((x1 + x2) / 2) / w, 0, 1)),
            float(np.clip(((y1 + y2) / 2) / h, 0, 1)),
            float(np.clip((x2 - x1) / w, 0, 1)),
            float(np.clip((y2 - y1) / h, 0, 1)),
        )

    @staticmethod
    def _bbox_contrast(u8: np.ndarray, bbox: tuple) -> float:
        h, w = u8.shape
        cx, cy, bw, bh = bbox
        px, py = int(cx * w), int(cy * h)
        bw_px, bh_px = max(1, int(bw * w)), max(1, int(bh * h))
        x1, y1 = max(0, px - bw_px // 2), max(0, py - bh_px // 2)
        x2, y2 = min(w, px + bw_px // 2), min(h, py + bh_px // 2)
        roi = u8[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        pad = max(6, int(max(bw_px, bh_px) * 0.6))
        ry1, ry2 = max(0, y1 - pad), min(h, y2 + pad)
        rx1, rx2 = max(0, x1 - pad), min(w, x2 + pad)
        ring = u8[ry1:ry2, rx1:rx2].astype(np.float32)
        ring[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = np.nan
        ring_mean = float(np.nanmean(ring)) if np.any(~np.isnan(ring)) else float(u8.mean())
        return float(roi.mean() - ring_mean)


class NegativeSampler:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.ct = CoordinateTransformer()

    def sample(
        self,
        series_uid: str,
        hu_volume: np.ndarray,
        nodules_df: pd.DataFrame,
        spacing: tuple,
        origin: tuple,
        qualities: list[SliceQuality],
        positive_z: set[int],
        lung_mask: np.ndarray,
    ) -> list[SampleRecord]:
        cfg = self.cfg
        valid_z = [q.z for q in qualities if q.is_valid]
        if not valid_z:
            return []

        excluded = set(positive_z)
        excluded |= self._nodule_exclusion_z(nodules_df, spacing, origin, hu_volume.shape[0])

        candidates = [z for z in valid_z if z not in excluded]
        if len(candidates) < cfg.min_negative_pool:
            self.logger.debug(
                "%s: insufficient negative pool (%d)", series_uid, len(candidates)
            )
            return []

        n_pos_proxy = max(1, len(positive_z))
        n_neg = int(n_pos_proxy * cfg.negative_ratio)
        n_neg = min(n_neg, len(candidates), cfg.max_negatives_per_scan)

        rng = random.Random(cfg.random_seed + hash(series_uid) % 10_000)
        chosen = rng.sample(candidates, n_neg)

        return [
            SampleRecord(
                series_uid=series_uid,
                slice_index=z,
                is_positive=False,
                bbox=None,
                lung_coverage=qualities[z].lung_coverage,
            )
            for z in chosen
        ]

    def _nodule_exclusion_z(
        self,
        nodules_df: pd.DataFrame,
        spacing: tuple,
        origin: tuple,
        z_max: int,
    ) -> set[int]:
        cfg = self.cfg
        excl = set()
        margin_vox = int(np.ceil(cfg.nodule_exclusion_mm / spacing[2]))
        for _, n in nodules_df.iterrows():
            _, _, vz = self.ct.world_to_voxel(
                (n["coordX"], n["coordY"], n["coordZ"]),
                spacing,
                origin,
            )
            zi = int(np.clip(vz, 0, z_max - 1))
            r_vox = int(np.ceil((float(n["diameter_mm"]) / 2 + cfg.nodule_exclusion_mm) / spacing[2]))
            for z in range(max(0, zi - r_vox - margin_vox), min(z_max, zi + r_vox + margin_vox + 1)):
                excl.add(z)
        return excl


# =============================================================================
# SPLIT & LEAKAGE
# =============================================================================


class PatientSplitter:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def assign_patients(self, patient_ids: list[str]) -> dict[str, str]:
        """Map each series_uid to train/val/test before any slice export."""
        cfg = self.cfg
        patients = sorted(set(patient_ids))
        train_p, temp_p = train_test_split(
            patients,
            test_size=cfg.val_ratio + cfg.test_ratio,
            random_state=cfg.random_seed,
        )
        val_frac = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
        val_p, test_p = train_test_split(
            temp_p, test_size=1 - val_frac, random_state=cfg.random_seed
        )
        mapping: dict[str, str] = {}
        for p in train_p:
            mapping[p] = "train"
        for p in val_p:
            mapping[p] = "val"
        for p in test_p:
            mapping[p] = "test"
        self.logger.info(
            "Patient assignment: train=%d val=%d test=%d (total %d)",
            len(train_p),
            len(val_p),
            len(test_p),
            len(patients),
        )
        return mapping

    def split(self, samples: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
        cfg = self.cfg
        by_patient: dict[str, list[SampleRecord]] = {}
        for s in samples:
            by_patient.setdefault(s.series_uid, []).append(s)

        patients = sorted(by_patient.keys())
        train_p, temp_p = train_test_split(
            patients,
            test_size=cfg.val_ratio + cfg.test_ratio,
            random_state=cfg.random_seed,
        )
        val_frac = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
        val_p, test_p = train_test_split(
            temp_p, test_size=1 - val_frac, random_state=cfg.random_seed
        )

        out = {"train": [], "val": [], "test": []}
        mapping = {"train": train_p, "val": val_p, "test": test_p}
        for split_name, plist in mapping.items():
            for p in plist:
                for s in by_patient[p]:
                    s.split = split_name
                    out[split_name].append(s)

        self.logger.info(
            "Split patients: train=%d val=%d test=%d | slices: train=%d val=%d test=%d",
            len(train_p),
            len(val_p),
            len(test_p),
            len(out["train"]),
            len(out["val"]),
            len(out["test"]),
        )
        return out


class LeakageVerifier:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def verify_patient_mapping(self, patient_to_split: dict[str, str]) -> bool:
        train_p = {p for p, s in patient_to_split.items() if s == "train"}
        val_p = {p for p, s in patient_to_split.items() if s == "val"}
        test_p = {p for p, s in patient_to_split.items() if s == "test"}
        leaks = []
        if train_p & val_p:
            leaks.append(f"train/val overlap: {len(train_p & val_p)}")
        if train_p & test_p:
            leaks.append(f"train/test overlap: {len(train_p & test_p)}")
        if val_p & test_p:
            leaks.append(f"val/test overlap: {len(val_p & test_p)}")
        if any("_neg" in p for p in patient_to_split):
            leaks.append("Found _neg suffix in patient IDs")
        if leaks:
            for L in leaks:
                self.logger.error("LEAKAGE: %s", L)
            return False
        self.logger.info("Leakage verification PASSED")
        return True

    def verify(self, splits: dict[str, list[SampleRecord]]) -> bool:
        train_p = {s.series_uid for s in splits["train"]}
        val_p = {s.series_uid for s in splits["val"]}
        test_p = {s.series_uid for s in splits["test"]}
        leaks = []
        if train_p & val_p:
            leaks.append(f"train/val overlap: {len(train_p & val_p)}")
        if train_p & test_p:
            leaks.append(f"train/test overlap: {len(train_p & test_p)}")
        if val_p & test_p:
            leaks.append(f"val/test overlap: {len(val_p & test_p)}")
        if any("_neg" in p for p in train_p | val_p | test_p):
            leaks.append("Found _neg suffix in patient IDs")

        if leaks:
            for L in leaks:
                self.logger.error("LEAKAGE: %s", L)
            return False
        self.logger.info("Leakage verification PASSED")
        return True


# =============================================================================
# STREAMING EXPORT (save each scan to disk immediately)
# =============================================================================


class StreamingSliceWriter:
    """
    Write PNG + YOLO label per slice as each scan is processed.
    Avoids holding all volumes or re-loading CT data for a second export pass.
    
    v2.5: Exports 2.5D RGB images (R=Z-gap, G=Z, B=Z+gap) for depth context.
    """

    METADATA_FIELDS = [
        "filename",
        "split",
        "series_uid",
        "slice_index",
        "is_positive",
        "lung_coverage",
        "diameter_mm",
        "bbox_contrast",
        "nodule_id",
    ]

    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.window = HUWindowing(cfg.window)
        self.root = cfg.output_dir
        self.seen_hashes: dict[str, set[str]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }
        self.split_counters: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.total_written = 0
        self.metadata_path = self.root / "metadata" / "dataset_metadata.csv"
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.METADATA_FIELDS).writeheader()

    def write_scan(
        self,
        split: str,
        hu_volume: np.ndarray,
        samples: list[SampleRecord],
        series_uid: str,
    ) -> int:
        """Save all slices for one scan; returns number of files written."""
        cfg = self.cfg
        img_dir = self.root / "images" / split
        lbl_dir = self.root / "labels" / split
        written = 0

        for rec in samples:
            # Create 2.5D RGB image or single slice based on config
            if cfg.use_25d_stacking:
                u8 = self._create_25d_image(hu_volume, rec.slice_index)
            else:
                u8 = self.window.apply(hu_volume[rec.slice_index])
            
            idx = self.split_counters[split]
            fname = f"{series_uid}_z{rec.slice_index:04d}_{idx:05d}"

            img_hash = hashlib.md5(u8.tobytes()).hexdigest()
            if img_hash in self.seen_hashes[split]:
                continue
            self.seen_hashes[split].add(img_hash)

            # Save as PNG (lossless for medical images)
            img_path = img_dir / f"{fname}.png"
            cv2.imwrite(str(img_path), u8)

            lbl_path = lbl_dir / f"{fname}.txt"
            if rec.is_positive and rec.bbox is not None:
                b = rec.bbox
                lbl_path.write_text(
                    f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n",
                    encoding="utf-8",
                )
            else:
                lbl_path.write_text("", encoding="utf-8")

            self._append_metadata(
                {
                    "filename": fname,
                    "split": split,
                    "series_uid": series_uid,
                    "slice_index": rec.slice_index,
                    "is_positive": rec.is_positive,
                    "lung_coverage": round(rec.lung_coverage, 6),
                    "diameter_mm": rec.diameter_mm,
                    "bbox_contrast": round(rec.bbox_contrast, 4),
                    "nodule_id": rec.nodule_id,
                }
            )
            self.split_counters[split] += 1
            self.total_written += 1
            written += 1

        return written
    
    def _create_25d_image(self, hu_volume: np.ndarray, z_center: int) -> np.ndarray:
        """
        Create 2.5D RGB stacked image from 3 slices.
        R = Z-gap, G = Z (center), B = Z+gap
        Uses float32 internally for precision, converts to uint8 at end.
        """
        cfg = self.cfg
        z_max = hu_volume.shape[0]
        gap = cfg.slice_gap_25d
        
        # Handle boundaries
        z_minus = max(0, z_center - gap)
        z_plus = min(z_max - 1, z_center + gap)
        
        # Apply HU windowing to all 3 slices (keep as float32)
        lo, hi = self.window.preset.hu_min, self.window.preset.hu_max
        
        slice_minus = np.clip(hu_volume[z_minus], lo, hi).astype(np.float32)
        slice_center = np.clip(hu_volume[z_center], lo, hi).astype(np.float32)
        slice_plus = np.clip(hu_volume[z_plus], lo, hi).astype(np.float32)
        
        # Normalize each slice to [0, 1]
        norm_minus = (slice_minus - lo) / max(hi - lo, 1e-6)
        norm_center = (slice_center - lo) / max(hi - lo, 1e-6)
        norm_plus = (slice_plus - lo) / max(hi - lo, 1e-6)
        
        # Stack as RGB and convert to uint8
        rgb_25d = np.stack([norm_minus, norm_center, norm_plus], axis=-1)
        rgb_25d = (rgb_25d * 255.0).astype(np.uint8)
        
        return rgb_25d

    def _append_metadata(self, row: dict) -> None:
        with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.METADATA_FIELDS).writerow(row)

    def load_metadata(self) -> pd.DataFrame:
        if self.metadata_path.exists():
            return pd.read_csv(self.metadata_path)
        return pd.DataFrame(columns=self.METADATA_FIELDS)


class QualityAssurance:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def validate_export(self, root: Path) -> dict[str, Any]:
        issues: list[str] = []
        stats = {"splits": {}, "empty_positive_labels": 0, "malformed_labels": 0}

        for split in ("train", "val", "test"):
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
            if not img_dir.exists():
                issues.append(f"Missing {img_dir}")
                continue
            imgs = {p.stem for p in img_dir.glob("*.png")}
            lbls = {p.stem for p in lbl_dir.glob("*.txt")}
            stats["splits"][split] = {
                "images": len(imgs),
                "labels": len(lbls),
                "positives": 0,
            }
            if imgs != lbls:
                issues.append(f"{split}: image/label mismatch")
            for lbl in lbl_dir.glob("*.txt"):
                txt = lbl.read_text().strip()
                if not txt:
                    continue
                stats["splits"][split]["positives"] += 1
                parts = txt.split()
                if len(parts) != 5:
                    stats["malformed_labels"] += 1
                else:
                    coords = list(map(float, parts[1:]))
                    if not all(0 <= c <= 1 for c in coords):
                        stats["malformed_labels"] += 1
                    if coords[2] < 0.005 or coords[3] < 0.005:
                        issues.append(f"Tiny bbox: {lbl.name}")

        report = {"passed": len(issues) == 0, "issues": issues, "stats": stats}
        if issues:
            for i in issues[:20]:
                self.logger.warning("QA: %s", i)
        else:
            self.logger.info("Export QA PASSED")
        return report


# =============================================================================
# DIAGNOSTICS & VISUALIZATION
# =============================================================================


class DiagnosticsReporter:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.audit_dir = cfg.output_dir / "audit"

    def record_processing_stats(self, stats: dict) -> None:
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        path = self.audit_dir / "processing_stats.json"
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    def generate_histograms(self, samples: list[SampleRecord]) -> None:
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        pos = [s for s in samples if s.is_positive]
        neg = [s for s in samples if not s.is_positive]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, key, title in zip(
            axes.flatten(),
            ["lung_coverage", "lung_coverage", "diameter_mm", "bbox_contrast"],
            ["Lung coverage (neg)", "Lung coverage (pos)", "Nodule diameter mm", "Bbox contrast"],
        ):
            if key == "diameter_mm":
                data_p = [s.diameter_mm for s in pos if s.diameter_mm > 0]
                ax.hist(data_p, bins=40, color="green", alpha=0.8)
            elif key == "bbox_contrast":
                data_p = [s.bbox_contrast for s in pos]
                ax.hist(data_p, bins=40, color="green", alpha=0.8)
            else:
                ax.hist([s.lung_coverage for s in neg], bins=40, alpha=0.6, label="neg", density=True)
                ax.hist([s.lung_coverage for s in pos], bins=40, alpha=0.6, label="pos", density=True)
                ax.legend()
            ax.set_title(title)
        plt.tight_layout()
        fig.savefig(self.audit_dir / "v2_distributions.png", dpi=150)
        plt.close(fig)

    def compare_with_legacy(self, v2_samples: list[SampleRecord]) -> dict:
        cfg = self.cfg
        legacy_csv = cfg.legacy_audit_csv
        comparison: dict[str, Any] = {"v2": self._summarize(v2_samples)}

        if legacy_csv.exists():
            old = pd.read_csv(legacy_csv)
            comparison["v1"] = {
                "total_slices": len(old),
                "positives": int(old["has_label"].sum()),
                "pct_good_lung": round(
                    100 * (old["category"] == "good_lung").mean(), 2
                )
                if "category" in old.columns
                else None,
                "mean_lung_fraction": round(float(old["lung_fraction"].mean()), 4),
                "mean_intensity": round(float(old["mean_intensity"].mean()), 2),
                "pct_soft_high": round(
                    100 * (old["soft_tissue_fraction"] >= 0.55).mean(), 2
                ),
            }
        else:
            comparison["v1"] = {"note": "legacy audit CSV not found"}

        v1 = comparison.get("v1", {})
        v2s = comparison["v2"]
        comparison["expected_impact"] = {
            "recall": (
                "↑ Moderate–High: visibility-only bboxes reduce false negatives from "
                "mislabeled adjacent slices; higher lung-specific contrast via windowing."
            ),
            "fppi": (
                "↓ High: negatives restricted to lung parenchyma slices; fewer "
                "diaphragm/mediastinum hard negatives confusing the detector."
            ),
            "label_quality": (
                "↑ High: ~{:.0f}% fewer positive slice labels vs v1 blind z±3 propagation "
                "(exact reduction depends on nodule size distribution)."
            ).format(
                max(
                    0,
                    100
                    * (
                        1
                        - v2s["positives"]
                        / max(v1.get("positives", v2s["positives"]), 1)
                    ),
                )
            ),
            "mAP50_95": "↑ Moderate: cleaner labels + domain-matched negatives should lift both AP50 and AP75.",
        }
        return comparison

    @staticmethod
    def _summarize(samples: list[SampleRecord]) -> dict:
        pos = [s for s in samples if s.is_positive]
        neg = [s for s in samples if not s.is_positive]
        return {
            "total_slices": len(samples),
            "positives": len(pos),
            "negatives": len(neg),
            "pos_neg_ratio": round(len(pos) / max(len(neg), 1), 3),
            "mean_lung_coverage_pos": round(
                float(np.mean([s.lung_coverage for s in pos])) if pos else 0, 4
            ),
            "mean_lung_coverage_neg": round(
                float(np.mean([s.lung_coverage for s in neg])) if neg else 0, 4
            ),
            "mean_bbox_contrast": round(
                float(np.mean([s.bbox_contrast for s in pos])) if pos else 0, 2
            ),
            "unique_patients": len({s.series_uid for s in samples}),
        }

    def write_comparison_report(self, comparison: dict, cfg: PipelineConfig) -> None:
        path = self.audit_dir / "V1_VS_V2_COMPARISON.md"
        lines = [
            "# LUNA16 Dataset v1 vs v2 Comparison",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Scientific preprocessing decisions (v2)",
            "",
            "### 1. Lung segmentation (HU -1024 to -320)",
            "Chest CT lung parenchyma occupies this HU band. Morphological closing/opening ",
            "removes noise; two largest 3D components approximate left/right lungs.",
            "",
            "### 2. Slice quality gates",
            f"- Min lung coverage: **{cfg.min_lung_coverage:.0%}**",
            f"- Max soft-tissue fraction: **{cfg.max_soft_tissue_fraction:.0%}**",
            f"- Rejects apex/diaphragm/mediastinum-dominated slices",
            "",
            "### 3. HU window: {name} (center={center}, width={width})".format(
                name=cfg.window.name,
                center=cfg.window.center,
                width=cfg.window.width,
            ),
            "Maps [{:.0f}, {:.0f}] HU to 8-bit. Improves nodule–parenchyma contrast vs v1 [-1000,400].".format(
                cfg.window.hu_min, cfg.window.hu_max
            ),
            "",
            "### 4. Visibility-aware bboxes",
            "For each nodule sphere, include slice z only if |dz| ≤ R. In-plane diameter ",
            "from cross-section geometry; skip low-contrast or extra-pulmonary centers.",
            "",
            "### 5. Smart negatives",
            f"Sampled only from valid lung slices, excluding ±{cfg.nodule_exclusion_mm}mm ",
            f"around each nodule. Ratio={cfg.negative_ratio}, max/scan={cfg.max_negatives_per_scan}.",
            "",
            "### 6. Patient splits",
            "Single `series_uid` per scan — no `_neg` artificial patients.",
            "",
            "## Quantitative comparison",
            "",
            "```json",
            json.dumps(comparison, indent=2),
            "```",
            "",
            "## Expected metric impact",
            "",
        ]
        for k, v in comparison.get("expected_impact", {}).items():
            lines.append(f"- **{k}**: {v}")
        path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info("Wrote %s", path)

    def write_preprocessing_report(
        self, cfg: PipelineConfig, comparison: dict, proc_stats: dict
    ) -> None:
        path = self.audit_dir / "PREPROCESSING_REPORT.md"
        v2 = comparison.get("v2", {})
        lines = [
            "# LUNA16 Preprocessing v2 — Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Output:** `{cfg.output_dir}`",
            "",
            "## Configuration snapshot",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| HU window | {cfg.window.name} (C={cfg.window.center}, W={cfg.window.width}) |",
            f"| Min lung coverage | {cfg.min_lung_coverage:.0%} |",
            f"| Negative ratio | {cfg.negative_ratio} |",
            f"| Nodule exclusion | {cfg.nodule_exclusion_mm} mm |",
            f"| Min bbox contrast | {cfg.min_bbox_local_contrast} |",
            f"| Random seed | {cfg.random_seed} |",
            "",
            "## Processing summary",
            "",
            f"- Scans processed: {proc_stats.get('scans_processed', 0)}",
            f"- Scans failed: {proc_stats.get('scans_failed', 0)}",
            f"- Positive slices: {proc_stats.get('positives', 0)}",
            f"- Negative slices: {proc_stats.get('negatives', 0)}",
            f"- v2 total (pre-export): {v2.get('total_slices', 0)}",
            f"- Pos/neg ratio: {v2.get('pos_neg_ratio', 'n/a')}",
            "",
            "See `V1_VS_V2_COMPARISON.md` for legacy comparison and expected metric impact.",
            "See `qa_report.json`, `bbox_qa_overlays.png`, `v2_distributions.png` for QA artifacts.",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")


def draw_bbox_overlay(u8: np.ndarray, bbox: tuple) -> np.ndarray:
    rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    h, w = u8.shape
    cx, cy, bw, bh = bbox
    px, py = int(cx * w), int(cy * h)
    bw_px, bh_px = int(bw * w), int(bh * h)
    x1, y1 = px - bw_px // 2, py - bh_px // 2
    x2, y2 = px + bw_px // 2, py + bh_px // 2
    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return rgb


class QAVisualizer:
    def __init__(self, cfg: PipelineConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def window_comparison(self, hu: np.ndarray, z: int, out_path: Path) -> None:
        presets = PipelineConfig.presets()
        fig, axes = plt.subplots(1, len(presets), figsize=(4 * len(presets), 4))
        for ax, (_, preset) in zip(axes, presets.items()):
            w = HUWindowing(preset)
            ax.imshow(w.apply(hu[z]), cmap="gray")
            ax.set_title(f"{preset.name}\n[{preset.hu_min:.0f},{preset.hu_max:.0f}]")
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def bbox_overlays_from_disk(self, dataset_root: Path, meta: pd.DataFrame, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pos = meta[meta["is_positive"]]  # Pandas idiomatic boolean indexing
        if len(pos) == 0:
            return
        picks = pos.sample(min(self.cfg.qa_overlay_samples, len(pos)), random_state=self.cfg.random_seed)

        cols = 4
        rows = int(np.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
        axes = np.atleast_2d(axes).flatten()

        for ax, (_, row) in zip(axes, picks.iterrows()):
            split = row["split"]
            fname = row["filename"]
            img_path = dataset_root / "images" / split / f"{fname}.png"
            lbl_path = dataset_root / "labels" / split / f"{fname}.txt"
            if not img_path.exists():
                continue
            u8 = np.array(Image.open(img_path).convert("L"))
            txt = lbl_path.read_text(encoding="utf-8").strip()
            if txt:
                bbox = tuple(map(float, txt.split()[1:5]))
                disp = draw_bbox_overlay(u8, bbox)
            else:
                disp = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            ax.set_title(
                f"{split} z={row['slice_index']} c={row.get('bbox_contrast', 0):.1f}",
                fontsize=8,
            )
            ax.axis("off")
        for ax in axes[len(picks) :]:
            ax.axis("off")
        fig.suptitle("v2 QA: exported slices with bboxes", fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / "bbox_qa_overlays.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class LUNA16PipelineV2:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(cfg.output_dir / "audit" / "preprocessing.log", cfg.log_level)
        self.loader = ScanLoader(cfg, self.logger)
        self.segmenter = LungSegmenter(cfg, self.logger)
        self.quality = SliceQualityAnalyzer(cfg)
        self.nodule_extractor = NoduleSliceExtractor(cfg, self.logger)
        self.neg_sampler = NegativeSampler(cfg, self.logger)
        self.splitter = PatientSplitter(cfg, self.logger)
        self.leakage = LeakageVerifier(self.logger)
        self.diagnostics = DiagnosticsReporter(cfg, self.logger)
        self.qa_viz = QAVisualizer(cfg, self.logger)
        self.processing_stats: dict[str, Any] = {}

    def load_annotations(self) -> pd.DataFrame:
        path = self.cfg.luna16_base_h / "annotations.csv"
        df = pd.read_csv(path)
        df = df.dropna(subset=["coordX", "coordY", "coordZ", "diameter_mm"])
        df = df[df["diameter_mm"] > 0]
        self.logger.info("Loaded %d annotations, %d scans", len(df), df["seriesuid"].nunique())
        return df

    def process_scan(
        self,
        series_uid: str,
        split: str,
        nodules_df: pd.DataFrame,
        writer: Optional[StreamingSliceWriter],
        window_viz_cache: dict[str, np.ndarray],
    ) -> list[SampleRecord]:
        loaded = self.loader.load_hu_volume(series_uid)
        if loaded is None:
            self.processing_stats["scans_failed"] += 1
            return []
        hu, spacing, origin, _ = loaded

        lung_mask = self.segmenter.segment(hu, spacing)

        # --- v4: Tight Lung ROI Crop (NO background wipeout) ---
        # Use lung mask ONLY to find the bounding box, then crop.
        # The chest wall pixels are KEPT to preserve juxtapleural nodules.
        lung_coords = np.where(lung_mask > 0)
        if len(lung_coords[0]) > 0:
            pad = 30  # 30px safety margin for juxtapleural nodules
            y_min = max(0, lung_coords[1].min() - pad)
            y_max = min(hu.shape[1], lung_coords[1].max() + pad)
            x_min = max(0, lung_coords[2].min() - pad)
            x_max = min(hu.shape[2], lung_coords[2].max() + pad)
        else:
            y_min, y_max = 0, hu.shape[1]
            x_min, x_max = 0, hu.shape[2]

        hu = hu[:, y_min:y_max, x_min:x_max]
        lung_mask = lung_mask[:, y_min:y_max, x_min:x_max]
        crop_offset = (int(x_min), int(y_min))
        # --- end v4 crop ---

        # Recompute quality gates on the CROPPED volume so metrics are consistent
        qualities = self.quality.analyze_volume(hu, lung_mask)
        rejected = sum(1 for q in qualities if not q.is_valid)
        self.processing_stats["slices_rejected"]["total_z"] = (
            self.processing_stats["slices_rejected"].get("total_z", 0) + rejected
        )

        scan_nodules = nodules_df[nodules_df["seriesuid"] == series_uid]
        positives, pos_z = self.nodule_extractor.extract(
            hu, lung_mask, scan_nodules, spacing, origin, qualities, crop_offset
        )
        negatives = self.neg_sampler.sample(
            series_uid, hu, scan_nodules, spacing, origin, qualities, pos_z, lung_mask
        )
        samples = positives + negatives

        if writer is not None and samples:
            n_written = writer.write_scan(split, hu, samples, series_uid)
            self.processing_stats["slices_written"] = (
                self.processing_stats.get("slices_written", 0) + n_written
            )

        self.processing_stats["scans_processed"] += 1
        self.processing_stats["positives"] += len(positives)
        self.processing_stats["negatives"] += len(negatives)

        if len(window_viz_cache) < self.cfg.window_comparison_scans:
            window_viz_cache[series_uid] = hu.copy()
        del hu, lung_mask
        gc.collect()
        return samples

    def run(self, max_scans: Optional[int] = None, export: bool = True) -> None:
        self.processing_stats = {
            "scans_processed": 0,
            "scans_failed": 0,
            "slices_rejected": defaultdict(int),
            "positives": 0,
            "negatives": 0,
        }

        cfg = self.cfg
        if export and max_scans is None:
            for sub in ("images", "labels"):
                p = cfg.output_dir / sub
                if p.exists():
                    shutil.rmtree(p)
                    self.logger.info("Cleaned prior export: %s", p)

        self.logger.info("=" * 80)
        self.logger.info("LUNA16 Preprocessing v2 — START")
        self.logger.info("Output: %s", cfg.output_dir)
        self.logger.info(
            "Window: %s center=%s width=%s",
            cfg.window.name,
            cfg.window.center,
            cfg.window.width,
        )
        self.logger.info(
            "2.5D Stacking: %s (gap=%s slices)",
            "ENABLED" if cfg.use_25d_stacking else "DISABLED",
            cfg.slice_gap_25d,
        )

        annotations = self.load_annotations()
        scans = list(annotations["seriesuid"].unique())
        if max_scans:
            scans = scans[:max_scans]
            self.logger.info("Limited to %d scans (dry-run)", max_scans)

        patient_to_split = self.splitter.assign_patients(scans)
        if not self.leakage.verify_patient_mapping(patient_to_split):
            self.logger.error("Aborting: patient split leakage detected")
            return

        writer: Optional[StreamingSliceWriter] = None
        if export:
            writer = StreamingSliceWriter(cfg, self.logger)
            self.logger.info(
                "Streaming export enabled — each scan written to images/labels immediately"
            )

        all_samples: list[SampleRecord] = []
        window_viz_cache: dict[str, np.ndarray] = {}

        for uid in tqdm(scans, desc="Processing+exporting scans"):
            split = patient_to_split[uid]
            scan_nodules = annotations[annotations["seriesuid"] == uid]
            all_samples.extend(
                self.process_scan(uid, split, scan_nodules, writer, window_viz_cache)
            )

        self.diagnostics.generate_histograms(all_samples)
        comparison = self.diagnostics.compare_with_legacy(all_samples)
        self.diagnostics.record_processing_stats(
            {**self.processing_stats, "slices_rejected": dict(self.processing_stats["slices_rejected"])}
        )
        self.diagnostics.write_comparison_report(comparison, cfg)
        self.diagnostics.write_preprocessing_report(
            cfg, comparison, dict(self.processing_stats)
        )

        if cfg.generate_window_comparisons and window_viz_cache:
            audit = cfg.output_dir / "audit"
            for i, uid in enumerate(list(window_viz_cache.keys())[: cfg.window_comparison_scans]):
                hu = window_viz_cache[uid]
                z = hu.shape[0] // 2
                self.qa_viz.window_comparison(
                    hu, z, audit / f"window_comparison_{i}_{uid[-8:]}.png"
                )
            window_viz_cache.clear()
            gc.collect()

        if not export:
            self.logger.info("export=False — stopping after diagnostics")
            return

        meta_dir = cfg.output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        # Note: StreamingSliceWriter already exports metadata to dataset_metadata.csv
        # This pre-split CSV is for diagnostics only (can be removed if disk space is tight)
        pd.DataFrame([asdict(s) for s in all_samples]).to_csv(
            meta_dir / "samples_pre_split.csv", index=False
        )

        metadata_df = writer.load_metadata() if writer else pd.DataFrame()
        self.logger.info("Total slices on disk: %d", len(metadata_df))

        qa = QualityAssurance(cfg, self.logger)
        qa_report = qa.validate_export(cfg.output_dir)
        (cfg.output_dir / "audit" / "qa_report.json").write_text(
            json.dumps(qa_report, indent=2), encoding="utf-8"
        )

        if len(metadata_df) > 0:
            self.qa_viz.bbox_overlays_from_disk(
                cfg.output_dir, metadata_df, cfg.output_dir / "audit"
            )

        self._write_dataset_yaml(metadata_df, qa_report)
        self.logger.info("=" * 80)
        self.logger.info(
            "PIPELINE v2 COMPLETE — %d slices saved under %s",
            len(metadata_df),
            cfg.output_dir,
        )

    def _write_dataset_yaml(self, meta: pd.DataFrame, qa: dict) -> None:
        cfg = self.cfg
        content = {
            "path": str(cfg.output_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": {0: "nodule"},
            "preprocessing_version": "v2",
            "window": {
                "name": cfg.window.name,
                "center": cfg.window.center,
                "width": cfg.window.width,
                "hu_min": cfg.window.hu_min,
                "hu_max": cfg.window.hu_max,
            },
            "statistics": {
                "total_metadata_rows": len(meta),
                "positives": int(meta["is_positive"].sum()) if len(meta) else 0,
                "qa": qa.get("stats", {}),
            },
        }
        with open(cfg.output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            yaml.dump(content, f, default_flow_style=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LUNA16 YOLO preprocessing v2")
    p.add_argument("--max-scans", type=int, default=None, help="Limit scans for dry-run")
    p.add_argument("--no-export", action="store_true", help="Diagnostics only")
    p.add_argument(
        "--window",
        choices=list(PipelineConfig.presets().keys()),
        default="lung_default",
    )
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()
    if args.output:
        cfg.output_dir = Path(args.output)
    presets = PipelineConfig.presets()
    if args.window in presets:
        cfg.window = presets[args.window]

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)  # Reproducibility for any PyTorch ops

    pipeline = LUNA16PipelineV2(cfg)
    pipeline.run(max_scans=args.max_scans, export=not args.no_export)


if __name__ == "__main__":
    main()
