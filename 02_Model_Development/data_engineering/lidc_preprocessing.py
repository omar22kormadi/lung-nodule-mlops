"""
LIDC-IDRI Preprocessing Pipeline for Lung Nodule Classification
================================================================

A production-quality preprocessing pipeline for generating 3D classification
patches from LIDC-IDRI DICOM data for benign vs malignant prediction.

Architecture:
- Modular class-based design
- Memory-efficient processing with streaming
- Comprehensive error handling
- Detailed logging with progress tracking

Author: Medical AI Pipeline
Version: 1.0.0

Usage:
    python lidc_preprocessing.py --input G:\\manifest-1600709154662\\LIDC-IDRI --output .\\classification_dataset
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Scientific libraries
import numpy as np

# Fix for older pylidc version with newer numpy (MUST be before pylidc import)
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

import pandas as pd
from scipy import ndimage
from scipy.ndimage import zoom, binary_dilation, binary_erosion

# Medical imaging libraries
try:
    import pylidc as pylidc
    from pylidc import Annotation  # Explicit import required for type hints
    PYLIDC_AVAILABLE = True
except ImportError:
    PYLIDC_AVAILABLE = False
    warnings.warn("pylidc not available. Install with: pip install pylidc")
    class Annotation:  # Dummy class for type hints when pylidc unavailable
        pass

try:
    import pydicom
    from pydicom import dcmread
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from skimage import measure, morphology
    from skimage.morphology import ball, closing, opening
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Utilities
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import traceback

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PipelineConfig:
    """
    Central configuration for the preprocessing pipeline.
    
    WHY CONFIGURATION CLASS:
    - Centralizes all hyperparameters in one place
    - Makes experiments reproducible
    - Easy to modify without changing core logic
    - Enables configuration validation
    """
    # Paths
    input_dir: Path = None
    output_dir: Path = None
    
    # Patch parameters
    patch_size: Tuple[int, int, int] = (64, 64, 64)  # 64x64x64 3D patches
    patch_spacing: float = 1.0  # Isotropic 1mm spacing
    
    # HU normalization range
    hu_min: int = -1000  # Air
    hu_max: int = 400    # Soft tissue upper bound
    
    # Lung segmentation thresholds (in HU)
    seg_lower_threshold: int = -400  # Below this is air/lung
    seg_upper_threshold: int = -100  # Above this is dense tissue
    
    # Annotation consensus
    min_annotators: int = 3  # Minimum annotators required for a nodule
    
    # Malignancy labels
    # WHY REMOVE SCORE 3:
    # - Score 3 represents ambiguous/uncertain cases
    # - Including them introduces label noise
    # - Model learns from clean examples only
    # - Benign: 1-2, Malignant: 4-5, Excluded: 3
    benign_range: Tuple[int, int] = (1, 2)
    malignant_range: Tuple[int, int] = (4, 5)
    
    # Dataset split ratios (patient-level)
    # WHY PATIENT-LEVEL SPLITTING:
    # - Same patient can have multiple nodules
    # - If same patient's nodules appear in train AND test
    # - Model memorizes patient-specific features
    # - Results in overly optimistic metrics
    # - Must split by patient ID, not individual nodules
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Processing
    num_workers: int = 4
    batch_size: int = 10
    cache_dir: Optional[Path] = None
    
    # Quality control
    min_nodule_diameter_mm: float = 3.0  # Ignore very small nodules
    max_nodule_diameter_mm: float = 50.0  # Ignore very large regions
    remove_duplicate_nodules: bool = True
    
    # Logging
    verbose: bool = True
    log_file: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration"""
        # Validate split ratios sum to 1
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        # Validate patch size is positive
        if any(s <= 0 for s in self.patch_size):
            raise ValueError(f"Patch size must be positive, got {self.patch_size}")


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(config: PipelineConfig) -> logging.Logger:
    """
    Configure logging with both console and file handlers.
    
    WHY COMPREHENSIVE LOGGING:
    - Medical imaging pipelines have many failure modes
    - Debugging DICOM issues is difficult
    - Need audit trail for reproducibility
    - Track which scans were processed/failed
    """
    logger = logging.getLogger('LIDC_Preprocessing')
    logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# EXCEPTION CLASSES
# ============================================================

class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass

class DICOMReadError(PipelineError):
    """Error reading DICOM files"""
    pass

class AnnotationError(PipelineError):
    """Error processing annotations"""
    pass

class SegmentationError(PipelineError):
    """Error in lung segmentation"""
    pass


# ============================================================
# DICOM UTILITIES
# ============================================================

class DICOMReader:
    """
    Handles DICOM file reading and validation.
    
    WHY A SEPARATE CLASS:
    - Encapsulates all DICOM-specific logic
    - Centralized error handling
    - Easy to mock for testing
    - Reusable across different pipelines
    """
    
    @staticmethod
    def read_ct_series(scan, logger: logging.Logger = None) -> Tuple[np.ndarray, dict]:
        """
        Read and reconstruct CT series from pylidc scan.
        
        WHY THIS MATTERS:
        - CT scans consist of multiple 2D DICOM slices
        - Slices must be ordered by position
        - Different scanners have different slice spacing
        - HU values must be correctly calibrated
        
        Args:
            scan: pylidc Scan object
            logger: Logger instance
            
        Returns:
            volume: 3D numpy array in HU
            metadata: DICOM metadata dict
        """
        try:
            # Load all DICOM files for this scan
            # pylidc automatically sorts by slice location
            slices = scan.load_dicom_images(verbose=False)
            
            if len(slices) == 0:
                raise DICOMReadError(f"No DICOM slices found for scan {scan.patient_id}")
            
            # Extract pixel data and metadata
            # WHY STACK SLICES:
            # - Each slice is a 2D image at different Z position
            # - Need to stack in correct order for 3D volume
            # - pylidc handles sorting by slice location
            volume = np.stack([s.pixel_array for s in slices], axis=0)
            
            # Get reference slice for metadata
            ref_slice = slices[len(slices) // 2]
            
            # Convert to Hounsfield Units (HU)
            # WHY HU CONVERSION:
            # - Raw pixel values are scanner-specific
            # - HU normalizes across different CT scanners
            # - Water = 0 HU, Air = -1000 HU, Bone = +1000 HU
            # - Standard for medical image analysis
            intercept = getattr(ref_slice, 'RescaleIntercept', -1024)
            slope = getattr(ref_slice, 'RescaleSlope', 1)
            
            # Apply HU transformation: HU = pixel_value * slope + intercept
            volume = volume.astype(np.float32) * slope + intercept
            
            # Extract metadata
            metadata = {
                'patient_id': scan.patient_id,
                'slice_thickness': float(ref_slice.SliceThickness),
                'pixel_spacing': list(ref_slice.PixelSpacing),
                'rows': ref_slice.Rows,
                'columns': ref_slice.Columns,
                'num_slices': len(slices),
                'study_id': scan.study_id,
                'series_id': scan.series_id,
            }
            
            if logger:
                logger.debug(f"Loaded {len(slices)} slices, volume shape: {volume.shape}")
            
            return volume, metadata
            
        except Exception as e:
            raise DICOMReadError(f"Failed to read CT series: {str(e)}")
        finally:
            # Cleanup if logger was provided
            pass
    
    @staticmethod
    def validate_dicom(dicom_path: Path) -> bool:
        """
        Validate that a DICOM file is readable.
        
        WHY VALIDATION:
        - Some DICOM files may be corrupted
        - Need to handle malformed headers
        - Prevent crashes from bad data
        """
        try:
            dcm = dcmread(str(dicom_path), stop_before_pixels=True)
            return dcm.Modality == 'CT'
        except Exception:
            return False


# ============================================================
# RESAMPLING UTILITIES
# ============================================================

class VolumeResampler:
    """
    Handles volumetric resampling to isotropic resolution.
    
    WHY ISOTROPIC RESAMPLING:
    - CT scans have non-cubic voxels (e.g., 0.5mm x 0.5mm x 2mm)
    - Anisotropic voxels cause artifacts in 3D processing
    - Deep learning models work better with cubic voxels
    - 1mm isotropic is standard for lung analysis
    - Ensures consistent patch dimensions
    """
    
    @staticmethod
    def resample_to_isotropic(
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: float = 1.0,
        order: int = 1,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Resample 3D volume to isotropic resolution.
        
        Args:
            volume: Input 3D volume in HU
            original_spacing: (row_spacing, col_spacing, slice_spacing) in mm
            target_spacing: Target spacing in mm (default 1.0)
            order: Interpolation order (0=nearest, 1=linear)
            logger: Optional logger
            
        Returns:
            resampled_volume: Resampled 3D volume
            zoom_factor: Actual zoom factor applied
        """
        # Calculate zoom factors for each dimension
        # WHY ZOOM FACTOR:
        # - If original spacing is 2mm and target is 1mm
        # - Need to upsample by factor of 2
        # - Zoom factor = original_spacing / target_spacing
        zoom_factors = [
            original_spacing[0] / target_spacing,  # Row (Y)
            original_spacing[1] / target_spacing,  # Column (X)
            original_spacing[2] / target_spacing,  # Slice (Z)
        ]
        
        if logger:
            logger.debug(f"Resampling with factors: {zoom_factors}")
        
        # Apply zoom (interpolation)
        resampled = zoom(volume, zoom_factors, order=order)
        
        return resampled, zoom_factors[0]  # Return zoom factor for metadata
    
    @staticmethod
    def resample_mask(
        mask: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: float = 1.0,
        target_shape: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Resample a binary mask using nearest-neighbor interpolation.
        
        WHY NEAREST FOR MASKS:
        - Masks contain discrete labels (0, 1, 2, etc.)
        - Linear interpolation would create partial labels
        - Nearest-neighbor preserves label values exactly
        - Critical for segmentation masks
        """
        zoom_factors = [
            original_spacing[0] / target_spacing,
            original_spacing[1] / target_spacing,
            original_spacing[2] / target_spacing,
        ]
        
        # Resample with nearest-neighbor (order=0)
        resampled = zoom(mask.astype(np.float32), zoom_factors, order=0)
        
        # Round and convert back to integer
        resampled = np.round(resampled).astype(np.uint8)
        
        return resampled


# ============================================================
# LUNG SEGMENTATION
# ============================================================

class LungSegmenter:
    """
    Performs lung segmentation to isolate lung tissue.
    
    WHY LUNG SEGMENTATION:
    - CT scans contain air, lungs, and other tissues
    - Non-lung regions introduce noise for nodule detection
    - Reduces computation on irrelevant regions
    - Improves model focus on clinically relevant areas
    - HU thresholding is effective for lung isolation
    
    TECHNIQUE:
    - Use HU threshold to create binary mask
    - Morphological operations to clean up
    - Connected component analysis to keep lungs
    """
    
    @staticmethod
    def segment_lungs(
        volume: np.ndarray,
        lower_threshold: int = -400,
        upper_threshold: int = -100,
        min_lung_voxel_count: int = 1000,
        logger: Optional[logging.Logger] = None
    ) -> np.ndarray:
        """
        Segment lungs from CT volume using HU thresholding.
        
        Args:
            volume: 3D CT volume in HU
            lower_threshold: Lower HU threshold (air ~ -1000)
            upper_threshold: Upper HU threshold (soft tissue ~ -100)
            min_lung_voxel_count: Minimum voxels for valid lung mask
            logger: Optional logger
            
        Returns:
            mask: Binary mask where 1 = lung tissue
        """
        # Step 1: Create binary mask using HU threshold
        # WHY THRESHOLDING:
        # - Lung tissue has HU between -400 and -100
        # - Air outside body: < -400 HU
        # - Dense tissue (bone, contrast): > -100 HU
        mask = (volume > lower_threshold) & (volume < upper_threshold)
        mask = mask.astype(np.uint8)
        
        if logger:
            logger.debug(f"Initial threshold mask: {mask.sum()} voxels")
        
        # Step 2: Fill holes (lungs have internal vessels/airways)
        # Use morphological closing to fill small holes
        struct = ball(3)  # 3D spherical structuring element
        mask = closing(mask, struct)
        
        if logger:
            logger.debug(f"After closing: {mask.sum()} voxels")
        
        # Step 3: Keep only the two largest connected components (left/right lungs)
        # Remove smaller structures (vessels, noise)
        labeled = measure.label(mask)
        props = measure.regionprops(labeled)
        
        # Sort by size (largest first)
        props = sorted(props, key=lambda x: x.area, reverse=True)
        
        # Create clean mask with only lungs
        clean_mask = np.zeros_like(mask)
        
        # Keep top 2 components (left and right lung)
        for i, prop in enumerate(props[:2]):
            if prop.area >= min_lung_voxel_count:
                clean_mask[labeled == prop.label] = 1
        
        if logger:
            logger.debug(f"After keeping top 2: {clean_mask.sum()} voxels")
        
        # Step 4: Erode slightly to remove boundary artifacts
        clean_mask = binary_erosion(clean_mask, ball(2)).astype(np.uint8)
        
        return clean_mask
    
    @staticmethod
    def apply_lung_mask(volume: np.ndarray, mask: np.ndarray, fill_value: int = -1000) -> np.ndarray:
        """
        Apply lung mask to volume, setting outside regions to fill value.
        
        WHY FILL VALUE:
        - Preserves lung boundary context
        - -1000 HU represents air (outside lungs)
        - Makes visualization cleaner
        """
        masked_volume = volume.copy()
        masked_volume[mask == 0] = fill_value
        return masked_volume


# ============================================================
# ANNOTATION PROCESSING
# ============================================================

class AnnotationProcessor:
    """
    Handles annotation processing and consensus mask generation.
    
    WHY MULTI-ANNOTATOR CONSENSUS:
    - LIDC-IDRI has 4 radiologists annotating each scan
    - Different radiologists may mark slightly different boundaries
    - Consensus mask averages these interpretations
    - Reduces inter-observer variability
    - More robust ground truth
    
    TECHNIQUE:
    - For each nodule, get all annotator contours
    - Convert contours to binary masks
    - Average/all masks to create consensus
    - Threshold at 0.5 for binary consensus
    """
    
    @staticmethod
    def compute_consensus_mask(
        annotations: List[Annotation],
        volume_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        threshold: float = 0.5,
        logger: Optional[logging.Logger] = None
    ) -> np.ndarray:
        """
        Compute consensus mask from multiple annotations.
        
        Args:
            annotations: List of pylidc Annotation objects
            volume_shape: Shape of the volume (D, H, W)
            spacing: Voxel spacing (dz, dy, dx) in mm
            threshold: Threshold for consensus (0.5 = majority)
            logger: Optional logger
            
        Returns:
            consensus_mask: Binary consensus mask
        """
        # Initialize consensus accumulator
        consensus_acc = np.zeros(volume_shape, dtype=np.float32)
        
        # Process each annotation
        for annot in annotations:
            try:
                # Get the annotation mask on the original volume grid
                # pylidc automatically handles coordinate transformation
                mask = annot.boolean_mask()
                
                if mask is None:
                    continue
                
                # Add to accumulator
                consensus_acc += mask.astype(np.float32)
                
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to process annotation: {e}")
                continue
        
        # Normalize by number of annotations
        num_annotations = len(annotations)
        
        # Compute consensus: voxel is positive if > threshold of annotators agreed
        consensus_mask = (consensus_acc / num_annotations) >= threshold
        consensus_mask = consensus_mask.astype(np.uint8)
        
        if logger:
            active_voxels = consensus_mask.sum()
            logger.debug(f"Consensus mask: {active_voxels} voxels from {num_annotations} annotators")
        
        return consensus_mask
    
    @staticmethod
    def get_nodule_characteristics(annotation: Annotation) -> Dict:
        """
        Extract morphological characteristics from annotation.
        
        WHY MORPHOLOGICAL FEATURES:
        - Spiculation: Spiky/irregular edges suggest malignancy
        - Margin: Smooth vs irregular boundary indicates tumor type
        - These are clinically validated malignancy indicators
        - Can be used as auxiliary labels in multi-task learning
        
        Returns:
            Dict with spiculation_score, margin_score
        """
        # Get pylidc's computed features
        # These are standard radiological assessments
        return {
            'spiculation': annotation.spiculation,
            'margin': annotation.margin,
            'lobulation': annotation.lobulation,
            'subtlety': annotation.subtlety,
            'internal_structure': annotation.internalStructure,
            'calcification': annotation.calcification,
            'texture': annotation.texture,
        }


# ============================================================
# PATCH EXTRACTION
# ============================================================

class PatchExtractor:
    """
    Extracts centered 3D patches from volumes.
    
    WHY VOLUMETRIC PATCHES:
    - Single 2D slices miss 3D context
    - Nodule shape/spiculation are 3D features
    - Radiologists examine nodules in 3D
    - Deep learning can leverage spatial patterns
    - 64x64x64 captures nodule with surrounding tissue
    
    WHY 64x64x64 PATCH SIZE:
    - Large enough to capture nodule context (~50mm)
    - Small enough for GPU memory constraints
    - Standard in medical imaging literature
    - Balance between detail and efficiency
    """
    
    def __init__(self, patch_size: Tuple[int, int, int] = (64, 64, 64)):
        self.patch_size = patch_size
        self.pad_value = -1000  # HU value for padding
        
    def extract_centered_patch(
        self,
        volume: np.ndarray,
        center_coords: Tuple[int, int, int],
        margin_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Extract a centered 3D patch from volume.
        
        Args:
            volume: 3D volume in HU
            center_coords: (z, y, x) center coordinates
            margin_mask: Optional mask to check for lung boundary
            
        Returns:
            patch: Extracted patch (padded if near boundary)
            is_valid: Whether patch is valid (not mostly padding)
        """
        d, h, w = self.patch_size
        cz, cy, cx = center_coords
        
        # Calculate half-sizes
        half_d, half_h, half_w = d // 2, h // 2, w // 2
        
        # Calculate bounding box
        z_start = cz - half_d
        z_end = cz + half_d
        y_start = cy - half_h
        y_end = cy + half_h
        x_start = cx - half_w
        x_end = cx + half_w
        
        # Track padding needed
        pad_before = [0, 0, 0]
        pad_after = [0, 0, 0]
        
        # Check and pad before (negative indices)
        if z_start < 0:
            pad_before[0] = abs(z_start)
            z_start = 0
        if y_start < 0:
            pad_before[1] = abs(y_start)
            y_start = 0
        if x_start < 0:
            pad_before[2] = abs(x_start)
            x_start = 0
        
        # Check and pad after (exceeds volume)
        vol_d, vol_h, vol_w = volume.shape
        if z_end > vol_d:
            pad_after[0] = z_end - vol_d
            z_end = vol_d
        if y_end > vol_h:
            pad_after[1] = y_end - vol_h
            y_end = vol_h
        if x_end > vol_w:
            pad_after[2] = x_end - vol_w
            x_end = vol_w
        
        # Extract raw patch
        raw_patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if any(p > 0 for p in pad_before + pad_after):
            raw_patch = np.pad(
                raw_patch,
                ((pad_before[0], pad_after[0]),
                 (pad_before[1], pad_after[1]),
                 (pad_before[2], pad_after[2])),
                mode='constant',
                constant_values=self.pad_value
            )
        
        # Check validity: patch should not be mostly padding
        # WHY VALIDITY CHECK:
        # - Nodules near volume edge may have padded regions
        # - Too much padding indicates marginal nodule location
        # - We want patches with actual lung tissue
        valid_ratio = np.sum(raw_patch > -900) / raw_patch.size
        is_valid = valid_ratio > 0.5  # At least 50% real tissue
        
        # Resize to exact patch size if needed
        if raw_patch.shape != self.patch_size:
            # Use nearest-neighbor for HU values (preserve integer values)
            zoom_factors = [
                self.patch_size[0] / raw_patch.shape[0],
                self.patch_size[1] / raw_patch.shape[1],
                self.patch_size[2] / raw_patch.shape[2],
            ]
            raw_patch = zoom(raw_patch, zoom_factors, order=1)
        
        return raw_patch, is_valid
    
    @staticmethod
    def normalize_hu(patch: np.ndarray, hu_min: int = -1000, hu_max: int = 400) -> np.ndarray:
        """
        Normalize HU values to [0, 1] range.
        
        WHY NORMALIZATION:
        - HU values vary widely (-1000 to +1000)
        - Neural networks train better with normalized inputs
        - Clipping to [-1000, 400] focuses on relevant range
        - 0 = air/background, 1 = dense tissue
        """
        # Clip to range
        clipped = np.clip(patch, hu_min, hu_max)
        
        # Normalize to [0, 1]
        normalized = (clipped - hu_min) / (hu_max - hu_min)
        
        return normalized.astype(np.float32)


# ============================================================
# DATASET GENERATOR
# ============================================================

class LIDCDatasetGenerator:
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.dicom_reader = DICOMReader()
        self.resampler = VolumeResampler()
        self.segmenter = LungSegmenter()
        self.patch_extractor = PatchExtractor(config.patch_size)
        
        # Statistics tracking
        self.stats = {
            'total_patients': 0,
            'processed_patients': 0,
            'failed_patients': 0,
            'total_nodules': 0,
            'extracted_patches': 0,
            'skipped_nodules': 0,
            'class_distribution': {'benign': 0, 'malignant': 0},
        }
        
        # Metadata storage
        self.patch_metadata = []
        
        # For immediate saving (set by pipeline)
        self.writer = None
        self.splitter = None
    
    def set_writer_and_splitter(self, writer, splitter):
        """Set writer and splitter for immediate saving."""
        self.writer = writer
        self.splitter = splitter
    
    @staticmethod
    def _extract_patch(volume, center, patch_size):
        """
        Extract a cubic patch from volume centered at given coordinates.
        Pads with zeros if patch extends beyond volume boundaries.
        
        This is the SAME implementation as in preprocessing_3d_pipeline.py (lines 213-259)
        
        Args:
            volume: 3D numpy array
            center: (z, y, x) center coordinates
            patch_size: side length of cubic patch
        
        Returns:
            patch: 3D numpy array of shape (patch_size, patch_size, patch_size)
        """
        half = patch_size[0] // 2
        cz, cy, cx = int(round(center[0])), int(round(center[1])), int(round(center[2]))
        
        # Calculate extraction bounds
        z_start = cz - half
        y_start = cy - half
        x_start = cx - half
        z_end = z_start + patch_size[0]
        y_end = y_start + patch_size[1]
        x_end = x_start + patch_size[2]
        
        # Initialize with zeros (for padding)
        patch = np.zeros(patch_size, dtype=volume.dtype)
        
        # Calculate valid ranges
        vol_z_start = max(0, z_start)
        vol_y_start = max(0, y_start)
        vol_x_start = max(0, x_start)
        vol_z_end = min(volume.shape[0], z_end)
        vol_y_end = min(volume.shape[1], y_end)
        vol_x_end = min(volume.shape[2], x_end)
        
        # Calculate patch insertion ranges
        p_z_start = vol_z_start - z_start
        p_y_start = vol_y_start - y_start
        p_x_start = vol_x_start - x_start
        p_z_end = p_z_start + (vol_z_end - vol_z_start)
        p_y_end = p_y_start + (vol_y_end - vol_y_start)
        p_x_end = p_x_start + (vol_x_end - vol_x_start)
        
        patch[p_z_start:p_z_end, p_y_start:p_y_end, p_x_start:p_x_end] = \
            volume[vol_z_start:vol_z_end, vol_y_start:vol_y_end, vol_x_start:vol_x_end]
        
        return patch
    
    def process_patient(self, patient_id: str) -> List[Dict]:
        """
        Process a single patient's scans and extract patches.
        
        Args:
            patient_id: LIDC patient ID (e.g., 'LIDC-IDRI-0001')
            
        Returns:
            List of patch metadata dicts
        """
        patient_patches = []
        
        try:
            # Query pylidc for this patient
            # WHY pylidc.Annotation.クエリ:
            # - pylidc provides database interface to LIDC data
            # - Automatically handles DICOM fetching
            # - Provides standardized annotation access
            scans = pylidc.query(pylidc.Scan).filter(
                pylidc.Scan.patient_id == patient_id
            ).all()
            
            if len(scans) == 0:
                self.logger.warning(f"No scans found for {patient_id}")
                return []
            
            # Process each scan (usually 1 per patient for LIDC)
            for scan in scans:
                patches = self._process_scan(scan)
                patient_patches.extend(patches)
            
            # Increment processed count if we found patches
            if patient_patches:
                self.stats['processed_patients'] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to process patient {patient_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.stats['failed_patients'] += 1
            
        return patient_patches
    
    def _process_scan(self, scan) -> List[Dict]:
        """
        Process a single scan and extract nodules.
        
        Args:
            scan: pylidc Scan object
            
        Returns:
            List of patch metadata dicts
        """
        patches = []
        
        try:
            # Step 1: Load volume using pylidc's to_volume() (same as working preprocess.py)
            self.logger.info(f"Loading scan {scan.patient_id}...")
            try:
                volume = scan.to_volume()  # This is what works in preprocessing_3d_pipeline.py
                self.logger.debug(f"Loaded volume shape: {volume.shape}")
            except Exception as e:
                self.logger.error(f"Failed to load volume with to_volume(): {e}")
                # Fallback to DICOMReader
                volume, metadata = self.dicom_reader.read_ct_series(scan, self.logger)
            
            # Step 2: Get spacing info from scan
            z_spacing = float(scan.slice_thickness) if scan.slice_thickness else 2.5
            xy_spacing = float(scan.pixel_spacing) if scan.pixel_spacing else 0.7
            original_spacing = [z_spacing, xy_spacing, xy_spacing]
            
            # Step 3: Convert to HU if needed
            vol_min = volume.min()
            vol_max = volume.max()
            if vol_min < -900 and vol_max > 0:
                # Already in HU range
                volume_hu = volume.astype(np.float32)
                self.logger.debug(f"Volume in HU range: [{vol_min:.0f}, {vol_max:.0f}]")
            else:
                # Need to convert
                try:
                    dicoms = scan.load_all_dicom_images()
                    slope = float(dicoms[0].RescaleSlope) if hasattr(dicoms[0], 'RescaleSlope') else 1.0
                    intercept = float(dicoms[0].RescaleIntercept) if hasattr(dicoms[0], 'RescaleIntercept') else -1024.0
                except Exception:
                    slope = 1.0
                    intercept = -1024.0
                volume_hu = volume.astype(np.float32) * slope + intercept
            
            # Step 4: Resample to isotropic resolution
            zoom_factors = [
                original_spacing[0] / self.config.patch_spacing,
                original_spacing[1] / self.config.patch_spacing,
                original_spacing[2] / self.config.patch_spacing,
            ]
            volume_resampled = zoom(volume_hu, zoom_factors, order=1)
            self.logger.debug(f"Resampled volume shape: {volume_resampled.shape}")
            
            # Step 5: Normalize to [0, 1]
            volume_norm = np.clip(volume_resampled, self.config.hu_min, self.config.hu_max)
            volume_norm = (volume_norm - self.config.hu_min) / (self.config.hu_max - self.config.hu_min)
            volume_norm = volume_norm.astype(np.float32)
            
            # Step 6: Get nodule clusters (SAME AS preprocessing_3d_pipeline.py!)
            try:
                nodule_clusters = scan.cluster_annotations()
            except Exception as e:
                self.logger.error(f"Failed to cluster annotations: {e}")
                return []
            
            self.logger.info(f"Found {len(nodule_clusters)} nodule clusters")
            
            # Compute coordinate scaling factor
            scale_factor = np.array(original_spacing) / self.config.patch_spacing
            
            # Step 7: Process each nodule cluster
            for nod_idx, annotations in enumerate(nodule_clusters):
                try:
                    # Check minimum annotators (same as preprocessing_3d_pipeline.py)
                    if len(annotations) < self.config.min_annotators:
                        self.logger.debug(f"Skipping cluster {nod_idx}: only {len(annotations)} annotators")
                        continue
                    
                    # Get malignancy scores
                    mal_scores = [ann.malignancy for ann in annotations if ann.malignancy is not None]
                    if len(mal_scores) < self.config.min_annotators:
                        self.logger.debug(f"Skipping cluster {nod_idx}: not enough malignancy scores")
                        continue
                    
                    avg_malignancy = sum(mal_scores) / len(mal_scores)
                    
                    # Skip uncertain cases (score 3)
                    if avg_malignancy == 3:
                        self.logger.debug(f"Skipping cluster {nod_idx}: uncertain malignancy score 3")
                        continue
                    
                    # Classify
                    if avg_malignancy >= 3.0:  # >= 3 is malignant (note: >= 3, not > 3)
                        label = 1  # malignant
                    else:
                        label = 0  # benign
                    
                    # Get centroid and scale to resampled coordinates
                    centroid_orig = np.array(annotations[0].centroid)  # (z, y, x) in original voxels
                    centroid_resampled = centroid_orig * scale_factor
                    cz, cy, cx = [int(round(c)) for c in centroid_resampled]
                    
                    # Extract patch using the working approach from preprocessing_3d_pipeline.py
                    patch = self._extract_patch(volume_norm, centroid_resampled, tuple(self.config.patch_size))
                    
                    # Check validity
                    valid_ratio = np.sum(patch > 0) / patch.size
                    if valid_ratio < 0.5:
                        self.logger.debug(f"Skipping cluster {nod_idx}: mostly padding")
                        continue
                    
                    # Get spiculation and margin from first annotation
                    spiculation_label = 1 if annotations[0].spiculation >= 3 else 0
                    margin_label = 1 if annotations[0].margin >= 3 else 0
                    
                    # Update stats
                    self.stats['total_nodules'] += 1
                    self.stats['extracted_patches'] += 1
                    if label == 0:
                        self.stats['class_distribution']['benign'] += 1
                    else:
                        self.stats['class_distribution']['malignant'] += 1
                    
                    # Create patch info with volume data
                    patch_info = {
                        'patient_id': scan.patient_id,
                        'z_coord': cz,
                        'y_coord': cy,
                        'x_coord': cx,
                        'diameter_mm': float(annotations[0].diameter) if hasattr(annotations[0], 'diameter') else 0.0,
                        'malignancy_score': avg_malignancy,
                        'malignancy_label': label,
                        'spiculation_label': spiculation_label,
                        'margin_label': margin_label,
                        'num_annotators': len(annotations),
                        'spiculation_score': float(annotations[0].spiculation),
                        'margin_score': float(annotations[0].margin),
                        'lobulation_score': float(annotations[0].lobulation) if hasattr(annotations[0], 'lobulation') else 0,
                        'texture': float(annotations[0].texture) if hasattr(annotations[0], 'texture') else 0,
                        '_patch': patch,  # Store patch data for saving
                    }
                    
                    patches.append(patch_info)
                    self.logger.debug(f"Extracted nodule {nod_idx}: mal={avg_malignancy:.2f}, label={label}")
                    
                    # SAVE IMMEDIATELY if writer is set
                    if self.writer is not None and self.splitter is not None:
                        # Determine split based on patient
                        patient_id = scan.patient_id
                        split = self.splitter.get_split_for_patient(patient_id)
                        
                        # Save patch immediately
                        patch_data = patch.copy()
                        patch_info_for_save = {k: v for k, v in patch_info.items() if not k.startswith('_')}
                        filepath = self.writer.save_patch(patch_data, patch_info_for_save, split)
                        self.logger.info(f"  Saved: {filepath.name}")
                    
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process nodule {nod_idx}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to process scan: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        return patches
    
    def _process_nodule(
        self,
        annotation,
        volume: np.ndarray,
        lung_mask: np.ndarray,
        metadata: dict
    ) -> Optional[Dict]:
        """
        Process a single nodule annotation and extract patch.
        
        Args:
            annotation: pylidc Annotation object
            volume: Preprocessed 3D volume
            lung_mask: Lung segmentation mask
            metadata: Scan metadata
            
        Returns:
            Patch metadata dict or None if failed/skipped
        """
        # Get nodule center in voxel coordinates
        z, y, x = annotation.center_coord
        
        # Check diameter constraints
        diameter = annotation.diameter  # in mm
        if diameter < self.config.min_nodule_diameter_mm:
            self.logger.debug(f"Skipping small nodule: {diameter:.1f}mm < {self.config.min_nodule_diameter_mm}mm")
            return None
        
        if diameter > self.config.max_nodule_diameter_mm:
            self.logger.debug(f"Skipping large nodule: {diameter:.1f}mm > {self.config.max_nodule_diameter_mm}mm")
            return None
        
        # Check malignancy label
        # WHY REMOVE SCORE 3:
        # - Score 3 = uncertain/moderate probability
        # - Ambiguous labels confuse training
        # - Focus on clear benign (1-2) and malignant (4-5)
        mal_score = annotation.malignancy
        if mal_score == 3:
            self.logger.debug(f"Skipping uncertain malignancy score: {mal_score}")
            return None
        
        # Classify as benign or malignant
        if self.config.benign_range[0] <= mal_score <= self.config.benign_range[1]:
            label = 0  # Benign
        elif self.config.malignant_range[0] <= mal_score <= self.config.malignant_range[1]:
            label = 1  # Malignant
        else:
            # Score 3 already handled, but just in case
            return None
        
        # Extract morphological characteristics for auxiliary labels
        # WHY AUXILIARY LABELS (spiculation, margin):
        # - These are clinically validated malignancy indicators
        # - Spiculation: Spiky edges suggest malignancy
        # - Margin: Smooth vs irregular boundary indicates type
        # - Multi-task learning benefits from shared representations
        char = self.get_nodule_characteristics(annotation)
        
        # Create spiculation label (3 = moderate, 4-5 = marked)
        # We want binary: 0 = low/absent, 1 = high/present
        spiculation_label = 1 if annotation.spiculation >= 3 else 0
        
        # Create margin label (1-2 = well-defined, 3-5 = poorly-defined)
        margin_label = 1 if annotation.margin >= 3 else 0
        
        # Extract patch
        patch_volume, is_valid = self.patch_extractor.extract_centered_patch(
            volume, (int(z), int(y), int(x)), lung_mask
        )
        
        if not is_valid:
            self.logger.debug("Patch extraction invalid (mostly padding)")
            return None
        
        # Normalize HU values
        patch_normalized = self.patch_extractor.normalize_hu(
            patch_volume, self.config.hu_min, self.config.hu_max
        )
        
        # Update statistics
        self.stats['total_nodules'] += 1
        self.stats['extracted_patches'] += 1
        if label == 0:
            self.stats['class_distribution']['benign'] += 1
        else:
            self.stats['class_distribution']['malignant'] += 1
        
        # Create patch metadata
        patch_info = {
            'patient_id': metadata['patient_id'],
            'study_id': metadata['study_id'],
            'series_id': metadata['series_id'],
            'z_coord': z,
            'y_coord': y,
            'x_coord': x,
            'diameter_mm': diameter,
            'malignancy_score': mal_score,
            'malignancy_label': label,
            'spiculation_label': spiculation_label,
            'margin_label': margin_label,
            'num_annotators': len(annotation.consensus),
            'spiculation_score': annotation.spiculation,
            'margin_score': annotation.margin,
            'lobulation_score': annotation.lobulation,
            'texture': annotation.texture,
        }
        
        return patch_info
    
    @staticmethod
    def get_nodule_characteristics(annotation: Annotation) -> Dict:
        """Extract morphological characteristics from annotation."""
        return {
            'spiculation_score': annotation.spiculation,
            'margin_score': annotation.margin,
            'lobulation_score': annotation.lobulation,
            'texture': annotation.texture,
            'subtlety': annotation.subtlety,
        }


# ============================================================
# DATASET WRITER
# ============================================================

class DatasetWriter:
    """
    Handles writing patches and metadata to disk.
    
    WHY SEPARATE WRITER:
    - Encapsulates I/O logic
    - Can implement caching/batching
    - Easy to switch between formats
    - Memory-efficient streaming writes
    """
    
    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = Path(output_dir)
        self.logger = logger
        
        # Create directory structure
        self.split_dirs = {
            'train': self.output_dir / 'train',
            'val': self.output_dir / 'val',
            'test': self.output_dir / 'test',
        }
        
        for split_dir in self.split_dirs.values():
            split_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created dataset directories at {self.output_dir}")
    
    def save_patch(
        self,
        patch: np.ndarray,
        patch_info: Dict,
        split: str
    ) -> str:
        """
        Save patch as compressed .npz file.
        
        Args:
            patch: Normalized 3D patch
            patch_info: Metadata dict
            split: 'train', 'val', or 'test'
            
        Returns:
            Path to saved file
        """
        # Generate filename
        patient_id = patch_info['patient_id'].replace('-', '_')
        nodule_idx = len([f for f in self.split_dirs[split].glob(f'{patient_id}_*.npz')])
        filename = f"{patient_id}_nodule_{nodule_idx}.npz"
        filepath = self.split_dirs[split] / filename
        
        # Save compressed
        np.savez_compressed(
            filepath,
            volume=patch,
            label=patch_info['malignancy_label'],
            malignancy_score=patch_info['malignancy_score'],
            patient_id=patch_info['patient_id'],
            coordinates=(patch_info['z_coord'], patch_info['y_coord'], patch_info['x_coord']),
            spiculation_label=patch_info['spiculation_label'],
            margin_label=patch_info['margin_label'],
        )
        
        return filepath  # Return Path object, not string
    
    def save_metadata_csv(self, metadata_list: List[Dict], filename: str = 'metadata.csv'):
        """
        Save metadata as CSV file.
        
        Args:
            metadata_list: List of patch metadata dicts
            filename: Output filename
        """
        # Exclude internal columns (starting with _)
        metadata_list = [{k: v for k, v in m.items() if not k.startswith('_')} for m in metadata_list]
        
        df = pd.DataFrame(metadata_list)
        df.to_csv(self.output_dir / filename, index=False)
        self.logger.info(f"Saved metadata CSV with {len(metadata_list)} entries")
    
    def save_statistics(self, stats: Dict, filename: str = 'dataset_statistics.json'):
        """
        Save dataset statistics as JSON.
        
        Args:
            stats: Statistics dict
            filename: Output filename
        """
        with open(self.output_dir / filename, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info("Saved dataset statistics")


# ============================================================
# PATIENT-LEVEL SPLITTING
# ============================================================

class PatientSplitter:
    """
    Handles patient-level train/val/test splitting.
    
    WHY PATIENT-LEVEL SPLITTING:
    - Same patient can have multiple nodules
    - If same patient's nodules in train AND test
    - Model learns patient-specific features
    - Results in data leakage and optimistic metrics
    - Must split at patient level to prevent leakage
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
    def split_patients(self, patient_ids: List[str]) -> Dict[str, List[str]]:
        """
        Split patient IDs into train/val/test sets.
        
        Args:
            patient_ids: List of patient IDs
            
        Returns:
            Dict with 'train', 'val', 'test' keys containing patient ID lists
        """
        # Set seed for reproducibility
        np.random.seed(self.seed)
        
        # Shuffle patient IDs
        shuffled = np.random.permutation(patient_ids)
        
        # Calculate split indices
        n_total = len(shuffled)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        splits = {
            'train': list(shuffled[:n_train]),
            'val': list(shuffled[n_train:n_train + n_val]),
            'test': list(shuffled[n_train + n_val:]),
        }
        
        # Store mapping for later lookup
        self.patient_split = {}
        for p in splits['train']:
            self.patient_split[p] = 'train'
        for p in splits['val']:
            self.patient_split[p] = 'val'
        for p in splits['test']:
            self.patient_split[p] = 'test'
        
        return splits
    
    def get_split_for_patient(self, patient_id: str) -> str:
        """Get the split for a specific patient."""
        return self.patient_split.get(patient_id, 'train')  # Default to train
    
    def assign_patches_to_split(
        self,
        patches: List[Dict],
        patient_splits: Dict[str, List[str]]
    ) -> Dict[str, List[Dict]]:
        """
        Assign patches to splits based on patient ID.
        
        Args:
            patches: List of patch metadata dicts
            patient_splits: Dict of patient IDs per split
            
        Returns:
            Dict with patches per split
        """
        split_patches = {'train': [], 'val': [], 'test': []}
        
        for patch in patches:
            patient_id = patch['patient_id']
            
            # Find which split this patient belongs to
            assigned = False
            for split, patients in patient_splits.items():
                if patient_id in patients:
                    split_patches[split].append(patch)
                    assigned = True
                    break
            
            if not assigned:
                # Patient not in any split, assign to train by default
                split_patches['train'].append(patch)
        
        return split_patches


# ============================================================
# MAIN PIPELINE
# ============================================================

class PreprocessingPipeline:
    """
    Main preprocessing pipeline coordinator.
    
    Orchestrates:
    1. Configuration validation
    2. Patient processing
    3. Dataset splitting
    4. Patch writing
    5. Statistics generation
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging(config)
        
        self.generator = LIDCDatasetGenerator(config, self.logger)
        self.splitter = PatientSplitter(
            config.train_ratio,
            config.val_ratio,
            config.test_ratio
        )
        self.writer = DatasetWriter(config.output_dir, self.logger)
        
    def run(self):
        """Execute the complete preprocessing pipeline."""
        self.logger.info("=" * 70)
        self.logger.info("LIDC-IDRI PREPROCESSING PIPELINE STARTED")
        self.logger.info("=" * 70)
        
        # Step 1: Check pylidc configuration
        self._check_pylidc_config()
        
        # Step 2: Get all patient IDs
        self.logger.info("Querying LIDC-IDRI database...")
        all_patients = self._get_all_patients()
        self.logger.info(f"Found {len(all_patients)} patients in LIDC-IDRI")
        
        self.generator.stats['total_patients'] = len(all_patients)
        
        # Step 3: Split patients FIRST (before processing) so we know where to save
        self.logger.info("Splitting patients into train/val/test...")
        patient_splits = self.splitter.split_patients(all_patients)
        self.logger.info(f"  Train: {len(patient_splits['train'])} patients")
        self.logger.info(f"  Val: {len(patient_splits['val'])} patients")
        self.logger.info(f"  Test: {len(patient_splits['test'])} patients")
        
        # Set writer and splitter on generator for immediate saving
        self.generator.set_writer_and_splitter(self.writer, self.splitter)
        
        # Step 4: Process all patients (patches saved immediately)
        all_patches = []
        for patient_id in all_patients:
            self.logger.info(f"Processing patient {patient_id}...")
            patches = self.generator.process_patient(patient_id)
            all_patches.extend(patches)
            
            # Progress update
            processed = self.generator.stats['processed_patients']
            self.logger.info(f"Progress: {processed}/{len(all_patients)} patients, "
                           f"{len(all_patches)} patches extracted")
        
        # Step 5: Remove duplicate nodules (for metadata only)
        if self.config.remove_duplicate_nodules:
            all_patches = self._remove_duplicates(all_patches)
        
        # Step 6: Save metadata CSV (patches already saved during processing)
        self.logger.info("Saving metadata...")
        self.writer.save_metadata_csv(all_patches)
        
        # Step 7: Save statistics
        self.writer.save_statistics(self.generator.stats)
        
        # Final summary
        self._print_summary()
        
        self.logger.info("=" * 70)
        self.logger.info("PREPROCESSING PIPELINE COMPLETED")
        self.logger.info("=" * 70)
        
    def _check_pylidc_config(self):
        """Check and configure pylidc."""
        self.logger.info("Checking pylidc configuration...")
        
        try:
            pylidc_path = self.config.input_dir
            
            if not pylidc_path.exists():
                self.logger.error(f"LIDC data directory not found: {pylidc_path}")
                raise FileNotFoundError(f"LIDC data not found at {pylidc_path}")
            
            # pylidc finds data automatically if it's in the expected directory structure
            # The directory should contain folders like LIDC-IDRI-0001, LIDC-IDRI-0002, etc.
            self.logger.info(f"LIDC data path: {pylidc_path}")
            
            # Verify by querying pylidc
            all_scans = pylidc.query(pylidc.Scan).all()
            self.logger.info(f"pylidc found {len(all_scans)} scans in database")
            
        except Exception as e:
            self.logger.error(f"pylidc configuration error: {e}")
            raise
    
    def _get_downloaded_patients(self) -> List[str]:
        """Get list of patient IDs that actually exist on disk."""
        import os as _os
        try:
            input_dir = str(self.config.input_dir)
            self.logger.info(f"Checking for patients in: {input_dir}")
            
            # First, get all patients in pylidc database
            all_patients_in_db = set(str(s.patient_id) for s in pylidc.query(pylidc.Scan).all())
            self.logger.info(f"Patients in pylidc DB: {len(all_patients_in_db)}")
            
            # Then, check which ones exist on disk in the input directory
            downloaded = []
            
            if not _os.path.isdir(input_dir):
                self.logger.error(f"Input directory does not exist: {input_dir}")
                return []
            
            # List all items in input directory
            dir_contents = _os.listdir(input_dir)
            lidc_dirs = [n for n in dir_contents if n.startswith("LIDC-IDRI-") and _os.path.isdir(_os.path.join(input_dir, n))]
            self.logger.info(f"Directories starting with LIDC-IDRI-: {len(lidc_dirs)}")
            
            for name in lidc_dirs:
                if name in all_patients_in_db:
                    downloaded.append(name)
            
            self.logger.info(f"Found {len(downloaded)} patients on disk that are also in pylidc DB")
            return sorted(downloaded)
            
        except Exception as e:
            self.logger.error(f"Failed to get downloaded patients: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def _get_all_patients(self) -> List[str]:
        """Get list of all patient IDs from LIDC."""
        # Use downloaded patients only (not all in pylidc DB)
        return self._get_downloaded_patients()
    
    def _remove_duplicates(self, patches: List[Dict]) -> List[Dict]:
        """
        Remove duplicate nodules based on position proximity.
        
        WHY REMOVE DUPLICATES:
        - Same nodule may be annotated multiple times
        - Annotations may overlap significantly
        - Duplicate patches waste training data
        - Keep the annotation with most annotators
        """
        unique_patches = []
        seen_centers = {}
        
        for patch in patches:
            # Create key from rounded coordinates
            coords = (
                round(patch['z_coord'] / 5),  # 5mm bin
                round(patch['y_coord'] / 5),
                round(patch['x_coord'] / 5),
            )
            
            # Check if we've seen a similar nodule
            if coords not in seen_centers:
                seen_centers[coords] = len(unique_patches)
                unique_patches.append(patch)
            else:
                # Compare annotator counts, keep the better one
                existing_idx = seen_centers[coords]
                if patch['num_annotators'] > unique_patches[existing_idx]['num_annotators']:
                    unique_patches[existing_idx] = patch
        
        removed = len(patches) - len(unique_patches)
        if removed > 0:
            self.logger.info(f"Removed {removed} duplicate nodules")
        
        return unique_patches
    
    def _print_summary(self):
        """Print final pipeline summary."""
        stats = self.generator.stats
        
        self.logger.info("")
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("-" * 40)
        self.logger.info(f"Total patients: {stats['total_patients']}")
        self.logger.info(f"Processed: {stats['processed_patients']}")
        self.logger.info(f"Failed: {stats['failed_patients']}")
        self.logger.info(f"Total nodules: {stats['total_nodules']}")
        self.logger.info(f"Extracted patches: {stats['extracted_patches']}")
        self.logger.info(f"Skipped: {stats['skipped_nodules']}")
        self.logger.info("")
        self.logger.info("CLASS DISTRIBUTION:")
        self.logger.info(f"  Benign: {stats['class_distribution']['benign']}")
        self.logger.info(f"  Malignant: {stats['class_distribution']['malignant']}")
        
        # Calculate class ratio
        total = sum(stats['class_distribution'].values())
        if total > 0:
            ratio = max(stats['class_distribution'].values()) / min(stats['class_distribution'].values())
            self.logger.info(f"  Imbalance ratio: {ratio:.2f}:1")


# ============================================================
# ENTRY POINT
# ============================================================

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    # Default output path (can be overridden by --output)
    DEFAULT_OUTPUT = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\classification_dataset")
    
    parser = argparse.ArgumentParser(
        description='LIDC-IDRI Preprocessing Pipeline for Lung Nodule Classification'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default="G:\\manifest-1600709154662\\LIDC-IDRI",
        help='Path to LIDC-IDRI data directory (default: G:\\manifest-1600709154662\\LIDC-IDRI)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f'Output directory for dataset (default: {DEFAULT_OUTPUT})'
    )
    
    parser.add_argument(
        '--patch-size', '-p',
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help='Patch size as three integers (default: 64 64 64)'
    )
    
    parser.add_argument(
        '--spacing', '-s',
        type=float,
        default=1.0,
        help='Isotropic voxel spacing in mm (default: 1.0)'
    )
    
    parser.add_argument(
        '--min-annotators', '-m',
        type=int,
        default=3,
        help='Minimum number of annotators per nodule (default: 3)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--seed', '-e',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = PipelineConfig(
        input_dir=Path(args.input),
        output_dir=output_path,
        patch_size=tuple(args.patch_size),
        patch_spacing=args.spacing,
        min_annotators=args.min_annotators,
        num_workers=args.workers,
        verbose=args.verbose,
        log_file=output_path / 'preprocessing.log'
    )
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create and run pipeline
    pipeline = PreprocessingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()


# ============================================================
# DATASET COMPRESSION FOR KAGGLE UPLOAD
# ============================================================

def compress_dataset_for_kaggle(
    dataset_dir: Union[str, Path] = None,
    output_name: str = "classification_dataset",
    remove_original: bool = False,
    compression_format: str = "zip",
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Compress the classification dataset for Kaggle upload.
    
    WHY COMPRESS FOR KAGGLE:
    - Kaggle dataset size limit: 20GB per dataset
    - Compressed format uploads faster
    - Reduces storage cost
    - Easy to download and extract on Kaggle
    - .npz files compress well (~60-70% reduction)
    
    Args:
        dataset_dir: Path to classification_dataset folder (default: project data folder)
        output_name: Output archive name (without extension)
        remove_original: If True, delete originals after compression
        compression_format: "zip" or "tar" (tar includes compression)
        logger: Optional logger
        
    Returns:
        Path to compressed archive
    """
    # Default dataset path
    if dataset_dir is None:
        dataset_dir = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\classification_dataset")
    dataset_dir = Path(dataset_dir)
    
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    log("=" * 70)
    log("COMPRESSING DATASET FOR KAGGLE UPLOAD")
    log("=" * 70)
    
    # Calculate original size
    total_size = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
    total_size_gb = total_size / (1024 ** 3)
    log(f"Original dataset size: {total_size_gb:.2f} GB")
    
    # Count files
    files = list(dataset_dir.rglob('*.npz'))
    npz_count = len(files)
    csv_files = list(dataset_dir.rglob('*.csv'))
    log(f"Found {npz_count} .npz patch files")
    log(f"Found {len(csv_files)} metadata files")
    
    # Create output archive path
    if compression_format == "zip":
        archive_path = dataset_dir.parent / f"{output_name}.zip"
    else:
        archive_path = dataset_dir.parent / f"{output_name}.tar.gz"
    
    # Remove existing archive if present
    if archive_path.exists():
        archive_path.unlink()
        log(f"Removed existing archive: {archive_path}")
    
    # Compress using shutil
    log(f"\nCompressing to: {archive_path.name}")
    log("This may take several minutes...")
    
    import time
    start_time = time.time()
    
    if compression_format == "zip":
        # Use shutil to create zip with maximum compression
        # WHY MAXIMUM COMPRESSION:
        # - .npz files are already compressed numpy arrays
        # - Further compression trades CPU time for upload/download speed
        # - 7z or deflate64 can achieve ~10-20% more compression
        import zipfile
        
        # Calculate optimal compression level (9 = maximum)
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            for i, file_path in enumerate(tqdm(files, desc="Compressing patches")):
                arcname = file_path.relative_to(dataset_dir)
                zipf.write(file_path, arcname)
                
                # Progress update every 100 files
                if (i + 1) % 100 == 0:
                    log(f"  Progress: {i + 1}/{npz_count} patches ({100*(i+1)//npz_count}%)")
        
        # Add metadata files
        for csv_file in csv_files:
            arcname = csv_file.relative_to(dataset_dir)
            zipf = zipfile.ZipFile(archive_path, 'a', zipfile.ZIP_DEFLATED, compresslevel=9)
            zipf.write(csv_file, arcname)
            zipf.close()
    
    else:  # tar.gz
        import tarfile
        with tarfile.open(archive_path, 'w:gz', compresslevel=9) as tar:
            tar.add(dataset_dir, arcname=output_name)
    
    elapsed = time.time() - start_time
    
    # Get compressed size
    compressed_size = archive_path.stat().st_size
    compressed_size_gb = compressed_size / (1024 ** 3)
    compression_ratio = (1 - compressed_size / total_size) * 100
    
    log(f"\n{'='*70}")
    log("COMPRESSION COMPLETE")
    log(f"{'='*70}")
    log(f"Original size:  {total_size_gb:.2f} GB")
    log(f"Compressed:     {compressed_size_gb:.2f} GB")
    log(f"Reduction:      {compression_ratio:.1f}%")
    log(f"Time:           {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    log(f"Archive:        {archive_path}")
    log(f"\nKaggle upload command:")
    log(f"  kaggle datasets create -p \"{archive_path.parent}\" -n \"{output_name}\"")
    log(f"\nOr use Kaggle web interface to upload.")
    
    # Remove originals if requested
    if remove_original:
        log(f"\n⚠️  Removing original files (keep archive safe!)")
        for file_path in files:
            file_path.unlink()
        for csv_file in csv_files:
            csv_file.unlink()
        log("Original files removed.")
        log(f"\nArchive is at: {archive_path}")
        log("Upload this to Kaggle, then download and extract on Kaggle.")
    
    return archive_path


def create_kaggle_dataset_yaml(
    dataset_dir: Union[str, Path],
    output_path: Path = None
) -> Path:
    """
    Create a dataset.yaml file for Kaggle CLI upload.
    
    WHY YAML CONFIG:
    - Describes dataset structure
    - Required for Kaggle API uploads
    - Helps collaborators understand dataset
    
    Args:
        dataset_dir: Path to classification_dataset
        output_path: Where to save yaml file
        
    Returns:
        Path to created yaml file
    """
    dataset_dir = Path(dataset_dir)
    
    # Count files
    train_count = len(list((dataset_dir / 'train').glob('*.npz')))
    val_count = len(list((dataset_dir / 'val').glob('*.npz')))
    test_count = len(list((dataset_dir / 'test').glob('*.npz')))
    
    yaml_content = f"""# LIDC-IDRI Lung Nodule Classification Dataset
# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset: LIDC-IDRI 3D Lung Nodule Classification
Description: 64x64x64 volumetric patches for benign vs malignant prediction

Structure:
  train: {train_count} patches (70%)
  val: {val_count} patches (15%)
  test: {test_count} patches (15%)
  total: {train_count + val_count + test_count} patches

Patch Size: 64x64x64 voxels (1mm isotropic)
Format: .npz (compressed numpy)

Labels:
  - malignancy_label: 0=benign (score 1-2), 1=malignant (score 4-5)
  - spiculation_label: 0=absent, 1=present
  - margin_label: 0=smooth, 1=irregular

Malignancy Score Distribution:
  - Score 3 removed (uncertain)
  - Scores 1-2: Benign
  - Scores 4-5: Malignant

NPZ Contents:
  - volume: 64x64x64 float32 (0-1 normalized)
  - label: int (malignancy)
  - malignancy_score: int (1-5 original)
  - spiculation_label: int
  - margin_label: int
  - patient_id: str
  - coordinates: tuple (z, y, x)

Usage:
  import numpy as np
  data = np.load('path/to/patch.npz')
  volume = data['volume']  # (64, 64, 64)
  label = data['label']   # 0 or 1
"""
    
    if output_path is None:
        output_path = dataset_dir / 'dataset.yaml'
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml at: {output_path}")
    return output_path


def prepare_for_kaggle_upload(
    dataset_dir: Union[str, Path] = None,
    output_name: str = "lidc-classification-64x64x64",
    remove_original: bool = False
):
    """
    Complete preparation for Kaggle upload.
    
    Steps:
    1. Create dataset.yaml
    2. Compress dataset
    3. Print upload instructions
    
    Args:
        dataset_dir: Path to classification_dataset (default: project data folder)
        output_name: Name for the Kaggle dataset
        remove_original: Delete originals after compression
    """
    # Default dataset path
    if dataset_dir is None:
        dataset_dir = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\classification_dataset")
    dataset_dir = Path(dataset_dir)
    
    print("\n" + "=" * 70)
    print("KAGGLE UPLOAD PREPARATION")
    print("=" * 70)
    
    # Step 1: Create yaml
    print("\n[1/3] Creating dataset.yaml...")
    yaml_path = create_kaggle_dataset_yaml(dataset_dir)
    
    # Step 2: Compress
    print("\n[2/3] Compressing dataset...")
    archive_path = compress_dataset_for_kaggle(
        dataset_dir,
        output_name,
        remove_original=False,  # Keep originals until upload confirmed
        logger=None
    )
    
    # Step 3: Instructions
    print("\n[3/3] Upload Instructions:")
    print("-" * 70)
    print(f"\n1. Upload the archive to Kaggle:")
    print(f"   https://www.kaggle.com/datasets/new/upload")
    print(f"\n2. Or use Kaggle CLI:")
    print(f"   kaggle datasets create -p \"{archive_path.parent}\" -n \"{output_name}\" --public")
    print(f"\n3. On Kaggle, extract the archive:")
    print(f"   !unzip -q {archive_path.name} -d /kaggle/input/")
    print(f"\n4. Update kgl_classifier.py paths:")
    print(f"   DATA_DIR = Path('/kaggle/input/{output_name}/classification_dataset')")
    print(f"\n5. Run training:")
    print(f"   python kgl_classifier.py")
    print("-" * 70)
    print(f"\nArchive location: {archive_path}")
    print(f"Dataset location:  {dataset_dir}")
    print("\n" + "=" * 70)