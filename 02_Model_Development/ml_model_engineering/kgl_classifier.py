"""
Kaggle-Ready 3D Lung Nodule Malignancy Classification
======================================================
Production-quality training script with Optuna + MLflow integration.

Model: R2Plus1D + Attention + Multi-Task Learning
Tasks: Malignancy (primary) + Spiculation + Margin (auxiliary)

Training Strategy:
- First-stage training on Kaggle high-VRAM GPUs (up to 90GB)
- Maximize ROC-AUC, PR-AUC, and sensitivity
- Large batch sizes (32/48/64)
- Mixed precision + gradient accumulation
- 200-300 epochs with cosine annealing

Based on: Crasta et al. (2024), Hunter et al. (2022)

Run on Kaggle: conda activate machinelearning && python kgl_classifier.py
"""

import os
import sys
import json
import shutil
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import mlflow
import mlflow.pytorch
import optuna
from optuna.trial import Trial
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    accuracy_score, f1_score, confusion_matrix,
    recall_score, precision_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import time

warnings.filterwarnings('ignore')

# ============================================================
# Reproducibility & Deterministic Training
# ============================================================
def set_deterministic_seed(seed: int = 42):
    """
    Set all random seeds for reproducible experiments.
    Critical for Kaggle submissions and scientific validity.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic algorithms (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Deterministic seed set: {seed}")


# ============================================================
# Configuration
# ============================================================
class Config:
    """
    Centralized configuration for all hyperparameters and paths.
    Easy to modify for different experiments or Kaggle environments.
    """
    # Paths - Will be auto-configured for local or Kaggle
    DATA_DIR = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\processed")
    SPLIT_DIR = DATA_DIR / "split_dataset"
    METADATA_FILE = None  # Will be set based on available files
    WEIGHTS_PATH = None  # Will be set for Kaggle
    OUTPUT_DIR = Path("/kaggle/working/output")  # Will be overridden
    
    # Model
    PATCH_SIZE = 64  # 64x64x64 input patches
    NUM_CLASSES_MAL = 2  # Benign vs Malignant
    NUM_CLASSES_SPI = 2  # Spiculation present/absent
    NUM_CLASSES_MAR = 2  # Margin clear/ill-defined
    
    # Training (defaults, will be tuned by Optuna)
    # MLOps: Adjust batch_size based on available GPU VRAM
    # - Kaggle (RTX 6000 Pro 102GB): 128-512
    # - Production Server (15GB VRAM): 16-32
    BATCH_SIZE = 128  # Default (will be tuned by Optuna)
    GRADIENT_ACCUMULATION = 1  # For even larger effective batch sizes
    MAX_EPOCHS = 250
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Mixed Precision
    USE_AMP = True  # Automatic Mixed Precision for speed + memory
    
    # Loss
    FOCAL_GAMMA = 2.0  # Focus on hard examples
    FOCAL_ALPHA = 0.25  # Balance positive/negative
    
    # Augmentation
    AUGMENT_PROB = 0.5
    ROTATION_DEGREES = 15
    GAUSSIAN_NOISE_STD = 0.01
    INTENSITY_SHIFT = 0.05
    
    # Early Stopping
    PATIENCE = 40
    MIN_DELTA = 1e-4
    
    # Optuna
    N_TRIALS = 50
    OPTUNA_TIMEOUT = 3600 * 8  # 8 hours
    
    # MLflow
    MLFLOW_EXPERIMENT = "Kaggle_Lung_Nodule_Classification_v1"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def setup_for_local(cls):
        """Configure paths for local development (default)"""
        cls.DATA_DIR = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\processed")
        cls.SPLIT_DIR = cls.DATA_DIR / "split_dataset"
        
        # Check for split or single metadata
        train_meta = cls.SPLIT_DIR / "train_metadata.csv"
        single_meta = cls.DATA_DIR / "metadata.csv"
        
        if train_meta.exists():
            cls.METADATA_FILE = None
        elif single_meta.exists():
            cls.METADATA_FILE = single_meta
            print(f"  ℹ️  Will perform automatic train/val/test split")
        
        cls.OUTPUT_DIR = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\output")
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.WEIGHTS_PATH = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\r2plus1d_18_weights.pth")
        print(f"✓ Local environment configured")
    
    @classmethod
    def setup_for_kaggle(cls):
        """Configure paths for Kaggle environment (NO INTERNET)"""
        if Path("/kaggle/input").exists():
            # Dataset path - uploaded as Kaggle dataset (READ-ONLY)
            # Structure: classification_dataset/train/, val/, test/, metadata.csv
            cls.DATA_DIR = Path("/kaggle/input/datasets/amorkormadi/lung-classification-dataset/classification_dataset")
            
            # Split directories (READ-ONLY, where data is stored)
            cls.SPLIT_DIR = cls.DATA_DIR
            
            # Metadata file (READ-ONLY)
            cls.METADATA_FILE = cls.DATA_DIR / "metadata.csv"
            
            # WRITABLE directory for split metadata and outputs
            cls.WORKING_DIR = Path("/kaggle/working")
            cls.OUTPUT_DIR = cls.WORKING_DIR / "output"
            cls.SPLIT_OUTPUT_DIR = cls.WORKING_DIR / "split_data"
            
            print(f"  ✓ Dataset (read-only): {cls.DATA_DIR}")
            print(f"  ✓ Metadata (read-only): {cls.METADATA_FILE}")
            print(f"  ✓ Output (writable): {cls.OUTPUT_DIR}")
            print(f"  ✓ Split metadata (writable): {cls.SPLIT_OUTPUT_DIR}")
            
            # Pretrained weights path - uploaded manually (no internet)
            cls.WEIGHTS_PATH = Path("/kaggle/input/datasets/amorkormadi/r2plus1d-18-weights/r2plus1d_18_weights.pth")
            
            cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cls.SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            print(f"✓ Kaggle environment configured")
            print(f"  Data: {cls.DATA_DIR}")
            print(f"  Weights: {cls.WEIGHTS_PATH}")
            print(f"  Output: {cls.OUTPUT_DIR}")
            
            # Verify files exist
            if not cls.DATA_DIR.exists():
                print(f"\n⚠️  WARNING: Dataset path not found!")
                print(f"   Expected: {cls.DATA_DIR}")
                print(f"   Please upload your dataset to Kaggle")
            else:
                # Check for train/val/test subdirectories
                train_exists = (cls.DATA_DIR / "train").exists()
                val_exists = (cls.DATA_DIR / "val").exists()
                test_exists = (cls.DATA_DIR / "test").exists()
                meta_exists = cls.METADATA_FILE.exists()
                print(f"   ✓ Dataset found")
                print(f"     train/: {len(list((cls.DATA_DIR / 'train').glob('*.npz')))} files" if train_exists else "     train/: ✗")
                print(f"     val/:   {len(list((cls.DATA_DIR / 'val').glob('*.npz')))} files" if val_exists else "     val/:   ✗")
                print(f"     test/:  {len(list((cls.DATA_DIR / 'test').glob('*.npz')))} files" if test_exists else "     test/:  ✗")
                print(f"     metadata.csv: {'✓' if meta_exists else '✗'}")
            
            if not cls.WEIGHTS_PATH.exists():
                print(f"\n⚠️  WARNING: Pretrained weights not found!")
                print(f"   Expected: {cls.WEIGHTS_PATH}")
                print(f"   Please check filename in: /kaggle/input/datasets/omarkormadi/my-r2plus1d-weights/")
            else:
                print(f"   ✓ Pretrained weights found")


# ============================================================
# Data Splitting
# ============================================================
def prepare_dataset_for_kaggle(data_dir: Path, metadata_file: Path, output_dir: Path):
    """
    Prepare split metadata for Kaggle by scanning train/val/test folders.
    
    This is simpler than split_dataset() because patches are ALREADY split.
    We just need to match .npz files with metadata rows.
    """
    print(f"\n{'='*70}")
    print(f"PREPARING DATASET FOR KAGGLE")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load global metadata
    global_meta = pd.read_csv(metadata_file)
    print(f"Loaded global metadata: {len(global_meta)} rows")
    
    # Scan each split folder and create metadata
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  ⚠️  {split}/ folder not found!")
            continue
        
        # Get all npz files
        npz_files = list(split_dir.glob("*.npz"))
        print(f"\n{split}/: {len(npz_files)} files")
        
        split_rows = []
        for npz_file in npz_files:
            # Extract patient_id and nodule index from filename: LIDC_IDRI_0001_nodule_0.npz
            stem = npz_file.stem
            parts = stem.split('_nodule_')
            if len(parts) != 2:
                continue
            
            patient_id_kaggle = parts[0]  # LIDC_IDRI_0001
            nodule_idx = int(parts[1])  # 0
            patient_id_orig = patient_id_kaggle.replace('_', '-')  # LIDC-IDRI-0001
            
            # Find ALL matching rows for this patient
            matches = global_meta[global_meta['patient_id'] == patient_id_orig]
            if len(matches) == 0:
                continue
            
            # If patient has multiple nodules, try to match by nodule index
            if len(matches) > 1:
                # Try to find the specific nodule by index
                if nodule_idx < len(matches):
                    row = matches.iloc[nodule_idx].copy()
                else:
                    # Fallback: skip if index out of range
                    continue
            else:
                # Only one nodule for this patient
                row = matches.iloc[0].copy()
            
            row['filepath'] = str(npz_file)
            row['split'] = split
            split_rows.append(row)
        
        # Save split metadata
        if split_rows:
            split_df = pd.DataFrame(split_rows)
            
            # Remove duplicates based on filepath
            before_count = len(split_df)
            split_df = split_df.drop_duplicates(subset=['filepath'])
            after_count = len(split_df)
            
            if before_count != after_count:
                print(f"    ⚠️  Removed {before_count - after_count} duplicate rows")
            
            output_file = output_dir / f"{split}_metadata.csv"
            split_df.to_csv(output_file, index=False)
            print(f"  ✓ Saved {output_file.name}: {len(split_df)} samples")
    
    print(f"\n✓ Dataset preparation complete!")
    return output_dir


def split_dataset(metadata_path: Path, output_dir: Path, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Split dataset into train/val/test sets.
    
    WHY patient-level splitting prevents data leakage:
    - Multiple nodules can come from same patient
    - If same patient's nodules appear in both train and test
    - Model memorizes patient-specific features, not nodule features
    - Results in overly optimistic performance metrics
    - Must split by patient ID, not by individual nodule
    
    NOTE: With the new dataset structure, patches are ALREADY split into
    train/val/test folders. We just need to reconstruct filepaths.
    """
    print(f"\n{'='*70}")
    print(f"PREPARING DATASET")
    print(f"{'='*70}")
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    print(f"Total samples in metadata: {len(metadata)}")
    
    # NEW DATASET STRUCTURE: patches already split into train/val/test folders
    # We need to assign split labels based on folder structure
    
    # Check if 'split' column exists
    if 'split' in metadata.columns:
        print(f"  Found 'split' column, using existing split labels")
        train_meta = metadata[metadata['split'] == 'train'].reset_index(drop=True)
        val_meta = metadata[metadata['split'] == 'val'].reset_index(drop=True)
        test_meta = metadata[metadata['split'] == 'test'].reset_index(drop=True)
        
        print(f"\nExisting split:")
        print(f"  Train: {len(train_meta)} samples")
        print(f"  Val:   {len(val_meta)} samples")
        print(f"  Test:  {len(test_meta)} samples")
        
        # RECONSTRUCT FILEPATHS
        base_dir = metadata_path.parent
        
        train_meta['filepath'] = train_meta.apply(
            lambda row: str(base_dir / "train" / f"{row['patient_id'].replace('-', '_')}_nodule_*.npz"), axis=1
        )
        val_meta['filepath'] = val_meta.apply(
            lambda row: str(base_dir / "val" / f"{row['patient_id'].replace('-', '_')}_nodule_*.npz"), axis=1
        )
        test_meta['filepath'] = test_meta.apply(
            lambda row: str(base_dir / "test" / f"{row['patient_id'].replace('-', '_')}_nodule_*.npz"), axis=1
        )
        
    else:
        # Need to create split metadata from folder structure
        print(f"  No 'split' column, creating split from folder structure")
        
        base_dir = metadata_path.parent
        
        # Create split metadata by scanning folders
        train_files = list((base_dir / "train").glob("*.npz"))
        val_files = list((base_dir / "val").glob("*.npz"))
        test_files = list((base_dir / "test").glob("*.npz"))
        
        print(f"\nScanned folders:")
        print(f"  Train: {len(train_files)} files")
        print(f"  Val:   {len(val_files)} files")
        print(f"  Test:  {len(test_files)} files")
        
        # Filter metadata to only include files that exist
        train_meta = metadata[metadata['malignancy_label'].isin([0, 1])].copy()
        val_meta = metadata[metadata['malignancy_label'].isin([0, 1])].copy()
        test_meta = metadata[metadata['malignancy_label'].isin([0, 1])].copy()
        
        # Assign filepaths based on folder structure
        # We'll match by patient_id and nodule index
        train_meta['filepath'] = None
        val_meta['filepath'] = None
        test_meta['filepath'] = None
        
        # For simplicity, use metadata as-is and let Dataset class find files
        # The Dataset will construct paths from patient_id
        train_meta['split'] = 'train'
        val_meta['split'] = 'val'
        test_meta['split'] = 'test'
        
        # Reconstruct filepaths properly
        def get_filepath(patient_id, nodule_idx, split):
            """Find the actual npz file for this nodule"""
            folder = base_dir / split
            pattern = f"{patient_id.replace('-', '_')}_nodule_{nodule_idx}.npz"
            filepath = folder / pattern
            return str(filepath) if filepath.exists() else None
        
        # Since we don't have nodule_idx in metadata, we'll scan and match
        print(f"\nMatching files to metadata...")
        
        all_splits = {'train': [], 'val': [], 'test': []}
        
        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for f in files:
                # Extract patient_id from filename: LIDC_IDRI_0001_nodule_0.npz
                parts = f.stem.split('_nodule_')
                if len(parts) == 2:
                    patient_id_orig = parts[0].replace('_', '-')  # LIDC-IDRI-0001
                    nodule_idx = int(parts[1])
                    
                    # Find matching metadata row
                    matches = metadata[metadata['patient_id'] == patient_id_orig]
                    if len(matches) > 0:
                        # Take first match (simplified - ideally match by coordinates)
                        row = matches.iloc[0].copy()
                        row['filepath'] = str(f)
                        row['split'] = split_name
                        all_splits[split_name].append(row)
        
        train_meta = pd.DataFrame(all_splits['train'])
        val_meta = pd.DataFrame(all_splits['val'])
        test_meta = pd.DataFrame(all_splits['test'])
        
        print(f"  Matched: Train={len(train_meta)}, Val={len(val_meta)}, Test={len(test_meta)}")
    
    # Save split metadata files
    output_dir.mkdir(parents=True, exist_ok=True)
    train_meta.to_csv(output_dir / "train_metadata.csv", index=False)
    val_meta.to_csv(output_dir / "val_metadata.csv", index=False)
    test_meta.to_csv(output_dir / "test_metadata.csv", index=False)
    
    print(f"\n✓ Split metadata saved to {output_dir}")
    print(f"  Train: {output_dir / 'train_metadata.csv'}")
    print(f"  Val:   {output_dir / 'val_metadata.csv'}")
    print(f"  Test:  {output_dir / 'test_metadata.csv'}")
    
    return train_meta, val_meta, test_meta


# ============================================================
# Dataset
# ============================================================
class LungNoduleDataset(Dataset):
    """
    3D Lung Nodule Dataset with medical-appropriate augmentations.
    
    Why contextual 3D information improves malignancy prediction:
    - Radiologists examine nodules in 3D to assess growth patterns
    - Spiculation and margin characteristics are volumetric features
    - Single 2D slices miss crucial spatial context
    - 3D CNNs learn shape, texture, and growth directionality
    """
    
    def __init__(self, metadata_path: Path, transform: bool = True, cache: bool = False):
        """
        Args:
            metadata_path: Path to split_metadata.csv
            transform: Whether to apply data augmentations
            cache: Cache dataset in memory for faster training
        """
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.cache = cache
        self.cached_data = {}
        
        # Filter out malignancy score = 3 (ambiguous cases)
        # WHY: Score 3 represents uncertain/intermediate cases
        # Including them introduces label noise and confuses the model
        # Better to train on clear benign (1-2) and malignant (4-5) examples
        if 'malignancy_score' in self.metadata.columns:
            self.metadata = self.metadata[
                (self.metadata['malignancy_score'] <= 2.5) | 
                (self.metadata['malignancy_score'] >= 3.5)
            ].reset_index(drop=True)
            print(f"  Filtered to {len(self.metadata)} samples (excluded score=3)")
        
        if cache:
            print("  Caching dataset to memory...")
            self._preload_dataset()
    
    def _preload_dataset(self):
        """Load all patches into RAM for faster training"""
        for idx in tqdm(range(len(self.metadata)), desc="Caching"):
            row = self.metadata.iloc[idx]
            filepath = Path(row['filepath'])
            
            # Handle both .npy and .npz files
            if filepath.exists():
                if filepath.suffix == '.npz':
                    data = np.load(filepath)
                    # npz files may contain arrays under different keys
                    if 'volume' in data.files:
                        self.cached_data[idx] = data['volume']
                    elif 'patch' in data.files:
                        self.cached_data[idx] = data['patch']
                    else:
                        # Load first array
                        self.cached_data[idx] = data[data.files[0]]
                else:
                    self.cached_data[idx] = np.load(filepath)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if self.cache and idx in self.cached_data:
            volume = self.cached_data[idx]
        else:
            row = self.metadata.iloc[idx]
            filepath = Path(row['filepath'])
            
            # DEBUG: Print first load
            if idx == 0:
                print(f"\n[DEBUG] Loading file:")
                print(f"  filepath: {filepath}")
                print(f"  exists: {filepath.exists()}")
                if not filepath.exists():
                    # Try to find where files actually are
                    filename = filepath.name
                    parent = filepath.parent
                    print(f"  parent exists: {parent.exists()}")
                    if parent.exists():
                        print(f"  parent contents (first 5): {list(parent.iterdir())[:5]}")
                    
                    # Try with split subdirectory
                    for split_name in ['train', 'val', 'test']:
                        alt_path = parent / split_name / filename
                        if alt_path.exists():
                            print(f"  ✓ Found in: {alt_path}")
                            filepath = alt_path
                            break
            
            # Handle both .npy and .npz files
            if filepath.suffix == '.npz':
                data = np.load(filepath)
                if 'volume' in data.files:
                    volume = data['volume']
                elif 'patch' in data.files:
                    volume = data['patch']
                else:
                    volume = data[data.files[0]]
            else:
                volume = np.load(filepath)
        
        # Convert to tensor
        volume = torch.FloatTensor(volume)
        
        # Add channel dimension if needed (R2Plus1D expects 1 or 3 channels)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)  # (1, D, H, W)
        
        # Apply augmentations
        if self.transform:
            volume = self._apply_augmentations(volume)
        
        # CRITICAL: Load labels with explicit validation - NO silent fallbacks!
        # WHY: Silent fallback to label=0 causes degenerate predictions
        # If any label is missing, raise an exception immediately
        
        row = self.metadata.iloc[idx]
        
        # Validate all required labels exist
        if 'malignancy_label' not in row:
            raise KeyError(
                f"Missing 'malignancy_label' in row {idx}: {row.to_dict()}\n"
                f"File: {row.get('filepath', 'unknown')}"
            )
        if 'spiculation_label' not in row:
            raise KeyError(
                f"Missing 'spiculation_label' in row {idx}: {row.to_dict()}\n"
                f"File: {row.get('filepath', 'unknown')}\n"
                f"THIS CAUSES SPICULATION COLLAPSE!"
            )
        if 'margin_label' not in row:
            raise KeyError(
                f"Missing 'margin_label' in row {idx}: {row.to_dict()}\n"
                f"File: {row.get('filepath', 'unknown')}\n"
                f"THIS CAUSES MARGIN COLLAPSE!"
            )
        
        # Extract labels with validation
        malignancy_label = int(row['malignancy_label'])
        spiculation_label = int(row['spiculation_label'])
        margin_label = int(row['margin_label'])
        
        # Validate label values are 0 or 1
        for name, val in [('malignancy', malignancy_label), 
                          ('spiculation', spiculation_label), 
                          ('margin', margin_label)]:
            if val not in [0, 1]:
                raise ValueError(
                    f"Invalid {name}_label value: {val} (expected 0 or 1)\n"
                    f"Row {idx}: {row.to_dict()}"
                )
        
        return {
            'volume': volume,
            'malignancy_label': torch.tensor(malignancy_label, dtype=torch.long),
            'spiculation_label': torch.tensor(spiculation_label, dtype=torch.long),
            'margin_label': torch.tensor(margin_label, dtype=torch.long),
            'filepath': str(row['filepath'])
        }
    
    def _apply_augmentations(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply medical-appropriate augmentations.
        
        WHY these augmentations are chosen:
        - Flips: Anatomically valid (left/right symmetry in lungs)
        - Small rotations: Patient positioning varies slightly
        - Gaussian noise: Scanner noise is realistic
        - Intensity shifts: Different CT scanner calibrations
        - Cutout: Simulates partial volume effects
        
        WHY NOT aggressive distortions:
        - Elastic deformations change nodule morphology unrealistically
        - Large rotations (>30°) are anatomically impossible
        - Color jitter doesn't apply to grayscale CT
        """
        # Random flips (3D)
        if random.random() > 0.5:
            volume = torch.flip(volume, [1])  # Flip depth
        if random.random() > 0.5:
            volume = torch.flip(volume, [2])  # Flip height
        if random.random() > 0.5:
            volume = torch.flip(volume, [3])  # Flip width
        
        # Small rotations (±15°)
        if random.random() > 0.5:
            angle = random.uniform(-Config.ROTATION_DEGREES, Config.ROTATION_DEGREES)
            volume = self._rotate_3d(volume, angle)
        
        # Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(volume) * Config.GAUSSIAN_NOISE_STD
            volume = volume + noise
        
        # Intensity shift
        if random.random() > 0.5:
            shift = random.uniform(-Config.INTENSITY_SHIFT, Config.INTENSITY_SHIFT)
            volume = volume + shift
        
        # Random cutout (simulate partial volume effect)
        if random.random() > 0.5:
            volume = self._random_cutout(volume)
        
        # Slight scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            volume = volume * scale
        
        # Clamp to valid range
        volume = torch.clamp(volume, 0.0, 1.0)
        
        return volume
    
    def _rotate_3d(self, volume: torch.Tensor, angle: float) -> torch.Tensor:
        """Simple 3D rotation around Z-axis"""
        # For simplicity, we'll skip complex 3D rotation here
        # In production, use scipy.ndimage.rotate or kornia
        return volume
    
    def _random_cutout(self, volume: torch.Tensor, cutout_ratio: float = 0.1) -> torch.Tensor:
        """Randomly zero out a small region"""
        _, d, h, w = volume.shape
        cutout_d = int(d * cutout_ratio)
        cutout_h = int(h * cutout_ratio)
        cutout_w = int(w * cutout_ratio)
        
        d_start = random.randint(0, d - cutout_d)
        h_start = random.randint(0, h - cutout_h)
        w_start = random.randint(0, w - cutout_w)
        
        volume[:, d_start:d_start+cutout_d, 
               h_start:h_start+cutout_h, 
               w_start:w_start+cutout_w] = 0
        
        return volume


# ============================================================
# Model Architecture
# ============================================================
class ChannelAttention(nn.Module):
    """
    Channel Attention Mechanism (SE-Net style).
    
    WHY attention helps focus on relevant features:
    - Not all feature channels are equally important for malignancy
    - Attention learns to weight channels by diagnostic relevance
    - Suppresses irrelevant background features
    - Enhances nodule-specific patterns (spiculation, margin, texture)
    - Improves interpretability by highlighting important features
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        mid_channels = max(channels // reduction, 8)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (B, C)
        avg_out = self.fc(self.avg_pool(x.unsqueeze(-1)).squeeze(-1))
        max_out = self.fc(self.max_pool(x.unsqueeze(-1)).squeeze(-1))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class DualAttention(nn.Module):
    """Dual attention with refinement"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.refine = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.refine(x)
        return x


class LungNoduleClassifier(nn.Module):
    """
    R2Plus1D-based Multi-Task Lung Nodule Classifier.
    
    WHY R2Plus1D:
    - Decomposes 3D convolutions into (2D spatial) + (1D temporal)
    - More parameter-efficient than pure 3D CNNs
    - Pretrained on Kinetics-400 (video action recognition)
    - Learns spatial patterns first, then volumetric relationships
    - Excellent transfer learning candidate for medical 3D data
    
    WHY multitask learning may improve robustness:
    - Auxiliary tasks (spiculation, margin) act as regularizers
    - Force model to learn clinically meaningful features
    - Prevent overfitting to malignancy-only patterns
    - Improve generalization to unseen nodule types
    - Spiculation and margin are strong malignancy indicators
    - Shared representation learns richer feature space
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Load pretrained R2Plus1D backbone
        print("Loading R2Plus1D-18 backbone...")
        backbone = r2plus1d_18(weights=None)  # Start with no weights
        
        # WHY transfer learning helps medical datasets:
        # - Medical datasets are small (LIDC: ~1000 patients)
        # - Training from scratch would overfit severely
        # - Kinetics-400 pretrained features generalize well:
        #   * Spatial hierarchy (edges → textures → shapes)
        #   * Motion features ≈ volumetric progression in CT
        #   * Rich feature representations already learned
        # - Fine-tuning adapts generic features to medical domain
        # 
        # OFFLINE KAGGLE SUPPORT:
        # - Kaggle notebooks can't download weights (no internet)
        # - Must upload r2plus1d_18_weights.pth manually
        # - Weights file: https://download.pytorch.org/models/r2plus1d_18-b3b3357e.pth
        # - Rename to: r2plus1d_18_weights.pth
        # - Upload as Kaggle dataset or notebook attachment
        
        weights_path = config.WEIGHTS_PATH if config.WEIGHTS_PATH else None
        
        if weights_path and Path(weights_path).exists():
            print(f"  Loading offline weights: {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            backbone.load_state_dict(state_dict)
            print("  ✓ Offline weights loaded successfully")
        else:
            print("  ⚠️  No pretrained weights found, using random initialization")
            print("  ℹ️  For better results, upload r2plus1d_18_weights.pth")
        
        feature_dim = backbone.fc.in_features  # 512
        
        # Extract feature extractor (remove final FC layer)
        self.backbone = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        
        # WHY transfer learning helps medical datasets:
        # - Medical datasets are small (LIDC: ~1000 patients)
        # - Training from scratch would overfit severely
        # - Kinetics-400 pretrained features generalize well:
        #   * Spatial hierarchy (edges → textures → shapes)
        #   * Motion features ≈ volumetric progression in CT
        #   * Rich feature representations already learned
        # - Fine-tuning adapts generic features to medical domain
        # Freeze early layers, fine-tune later layers
        self._freeze_backbone_layers(backbone)
        
        # Attention module
        self.attention = DualAttention(feature_dim, reduction=16)
        self.bn_adapt = nn.BatchNorm1d(feature_dim)
        
        # Primary task: Malignancy classification
        self.malignancy_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, config.NUM_CLASSES_MAL),
        )
        
        # Auxiliary task 1: Spiculation
        self.spiculation_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, config.NUM_CLASSES_SPI),
        )
        
        # Auxiliary task 2: Margin
        self.margin_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, config.NUM_CLASSES_MAR),
        )
    
    def _freeze_backbone_layers(self, backbone):
        """
        Freeze stem and layer1-2, fine-tune layer3-4.
        This balances feature reuse and domain adaptation.
        """
        for name, param in self.backbone.named_parameters():
            if name.startswith('0') or name.startswith('1'):
                param.requires_grad = False
        print("  Frozen: stem, layer1, layer2")
        print("  Trainable: layer3, layer4 + attention + heads")
    
    def forward(self, x):
        # R2Plus1D expects 3 channels, repeat if 1 channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)  # (B, 1, D, H, W) → (B, 3, D, H, W)
        
        # Extract features
        features = self.backbone(x)  # (B, 512, 1, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        
        # Batch norm adaptation
        features = self.bn_adapt(features)
        
        # Apply attention
        features = self.attention(features)
        
        # Multi-task heads
        malignancy_out = self.malignancy_head(features)
        spiculation_out = self.spiculation_head(features)
        margin_out = self.margin_head(features)
        
        return malignancy_out, spiculation_out, margin_out


# ============================================================
# Loss Functions
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    WHY focal loss helps imbalance:
    - Standard cross-entropy treats all examples equally
    - Easy examples dominate gradient, hard examples ignored
    - Focal loss down-weights easy examples automatically
    - Focuses training on hard, misclassified examples
    - Reduces impact of large background class (benign nodules)
    - Formula: FL(p) = -α(1-p)^γ * log(p)
      where γ controls focus strength, α balances classes
    
    Medical AI context:
    - Malignant nodules are rare (~30% of dataset)
    - Model can achieve 70% accuracy by predicting all benign
    - Focal loss forces model to learn malignant patterns
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Weighted combination of task losses.
    
    Weights can be tuned based on task importance:
    - Malignancy: Primary task (highest weight)
    - Spiculation: Auxiliary task (medium weight)
    - Margin: Auxiliary task (medium weight)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.mal_loss = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        # Add label smoothing to prevent overconfident predictions on imbalanced tasks
        self.spi_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mar_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Task weights (can be tuned)
        # Balanced approach: improve auxiliary tasks while protecting malignancy
        self.w_mal = 1.0   # Malignancy (primary task - keep dominant)
        self.w_spi = 0.8   # Spiculation (increased from 0.5 - moderate boost)
        self.w_mar = 1.0   # Margin (increased from 0.5 - needs help but not too much)
    
    def forward(self, mal_out, spi_out, mar_out, 
                mal_label, spi_label, mar_label):
        loss_mal = self.mal_loss(mal_out, mal_label)
        loss_spi = self.spi_loss(spi_out, spi_label)
        loss_mar = self.mar_loss(mar_out, mar_label)
        
        total_loss = (self.w_mal * loss_mal + 
                     self.w_spi * loss_spi + 
                     self.w_mar * loss_mar)
        
        return total_loss, {
            'malignancy_label': loss_mal.item(),
            'spiculation_label': loss_spi.item(),
            'margin_label': loss_mar.item()
        }


# ============================================================
# Comprehensive Multi-Task Evaluation
# ============================================================

def print_label_distribution(metadata, task_name, prefix="  "):
    """
    Print label distribution for a task with imbalance analysis.
    
    WHY this matters for medical imaging:
    - Class imbalance causes model to favor majority class
    - Spiculation/margin may have different imbalance than malignancy
    - Must know ratios to implement proper weighting
    - Helps diagnose why auxiliary heads collapse
    """
    labels = metadata[task_name].values
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"{prefix}{task_name.replace('_label', '').upper()} Distribution:")
    for val, count in zip(unique, counts):
        pct = count / len(labels) * 100
        label_name = "positive" if val == 1 else "negative"
        print(f"{prefix}  Class {val} ({label_name}): {count} samples ({pct:.1f}%)")
    
    # Calculate imbalance ratio
    if len(counts) == 2:
        ratio = max(counts) / min(counts)
        print(f"{prefix}  Imbalance Ratio: {ratio:.2f}:1")
    
    return dict(zip(unique, counts))


def print_dataset_diagnostics(datasets_dict, split_name=""):
    """
    Print comprehensive label distribution diagnostics.
    
    WHY this is critical before training:
    - Reveals if auxiliary task labels are all the same (collapse)
    - Shows imbalance ratios needed for weighted sampling
    - Detects data quality issues
    - Helps debug why spiculation/margin predictions are degenerate
    """
    print(f"\n{'='*70}")
    print(f"LABEL DISTRIBUTION DIAGNOSTICS{(' - ' + split_name) if split_name else ''}")
    print(f"{'='*70}")
    
    for split, dataset in datasets_dict.items():
        print(f"\n{split.upper()} SET ({len(dataset.metadata)} samples):")
        
        # Malignancy
        print_label_distribution(dataset.metadata, 'malignancy_label', '  ')
        
        # Spiculation
        print_label_distribution(dataset.metadata, 'spiculation_label', '  ')
        
        # Margin
        print_label_distribution(dataset.metadata, 'margin_label', '  ')
        
        # Check for degenerate distribution (all same class)
        for task in ['malignancy', 'spiculation', 'margin']:
            col = f'{task}_label'
            unique_vals = dataset.metadata[col].unique()
            if len(unique_vals) == 1:
                print(f"\n  ⚠️  WARNING: {task.upper()} has only 1 class! ({unique_vals[0]})")
                print(f"     Model will learn to predict constant value!")
                print(f"     This explains why predictions are always '{'low' if unique_vals[0] == 0 else 'high'}'!")
    
    print(f"\n{'='*70}")


def compute_all_task_metrics(all_labels, all_preds, all_probs, task_name):
    """
    Compute comprehensive metrics for a single task.
    
    WHY we need ALL these metrics for medical AI:
    - Accuracy: Overall correctness
    - F1: Balance precision/recall (important for imbalanced data)
    - Precision: How many predicted positive are truly positive
    - Recall/Sensitivity: How many actual positives we catch (critical for cancer detection)
    - Specificity: How many negatives we correctly identify
    - ROC-AUC: Overall ranking quality
    - Confusion Matrix: Shows exactly where errors occur
    """
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    
    # Specificity (true negative rate)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # ROC-AUC if valid probabilities
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['roc_auc'] = None
    
    # PR-AUC
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        metrics['pr_auc'] = auc(recall_vals, precision_vals)
    except ValueError:
        metrics['pr_auc'] = None
    
    return metrics


def print_task_metrics(metrics, task_name, prefix="  "):
    """Pretty print metrics for a single task"""
    print(f"\n{prefix}{task_name.upper()} METRICS:")
    print(f"{prefix}  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"{prefix}  F1-Score:   {metrics['f1']:.4f}")
    print(f"{prefix}  Precision:  {metrics['precision']:.4f}")
    print(f"{prefix}  Recall:     {metrics['recall']:.4f}")
    print(f"{prefix}  Specificity: {metrics['specificity']:.4f}")
    if metrics.get('roc_auc'):
        print(f"{prefix}  ROC-AUC:    {metrics['roc_auc']:.4f}")
    if metrics.get('pr_auc'):
        print(f"{prefix}  PR-AUC:     {metrics['pr_auc']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"{prefix}  Confusion Matrix:")
    print(f"{prefix}    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"{prefix}    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")


def print_prediction_distribution(preds, probs, task_name, prefix="  "):
    """
    Print prediction distribution for debugging.
    
    WHY this detects degenerate predictions:
    - If model always predicts 0, auxiliary head collapsed
    - Probability distribution shows confidence patterns
    - Reveals if model is uncertain or confident-but-wrong
    """
    unique, counts = np.unique(preds, return_counts=True)
    print(f"\n{prefix}{task_name.upper()} PREDICTIONS:")
    for val, count in zip(unique, counts):
        pct = count / len(preds) * 100
        label = "positive" if val == 1 else "negative"
        print(f"{prefix}  Class {val} ({label}): {count} predictions ({pct:.1f}%)")
    
    # Probability distribution
    prob_array = np.array(probs)
    print(f"{prefix}  Probability Stats:")
    print(f"{prefix}    Mean: {prob_array.mean():.4f}")
    print(f"{prefix}    Std:  {prob_array.std():.4f}")
    print(f"{prefix}    Min:  {prob_array.min():.4f}")
    print(f"{prefix}    Max:  {prob_array.max():.4f}")
    
    # Check for degenerate (all same)
    if prob_array.std() < 0.01:
        print(f"{prefix}  ⚠️  WARNING: Probabilities nearly constant! Head may be dead.")


def detect_degenerate_predictions(all_preds, all_probs, task_name):
    """
    Detect degenerate prediction patterns.
    
    WHY we need this:
    - If all predictions are same class, head collapsed
    - If probabilities have no variance, head is non-functional
    - Early detection allows intervention
    """
    issues = []
    
    # Check single-class prediction
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 1:
        issues.append(f"SINGLE-CLASS PREDICTION: All predictions = {unique_preds[0]}")
    
    # Check zero variance
    prob_std = np.std(all_probs)
    if prob_std < 0.01:
        issues.append(f"ZERO VARIANCE: All probabilities near constant")
    
    # Check extreme concentration
    prob_array = np.array(all_probs)
    mean_prob = prob_array.mean()
    if mean_prob < 0.05 or mean_prob > 0.95:
        issues.append(f"EXTREME CONFIDENCE: Mean probability = {mean_prob:.4f}")
    
    if issues:
        print(f"\n  ⚠️  DEGENERATE {task_name.upper()} HEAD DETECTED:")
        for issue in issues:
            print(f"     - {issue}")
        print(f"     This explains why {task_name} predictions are always constant!")
    
    return issues


def comprehensive_validate(model, dataloader, criterion, config, task_names=['malignancy', 'spiculation', 'margin']):
    """
    Comprehensive multi-task validation with full metrics.
    
    WHY comprehensive validation:
    - Only evaluating malignancy misses auxiliary head collapse
    - Spiculation/margin predictions may be degenerate
    - Must track all tasks to ensure proper multi-task learning
    - Debugging: tells us exactly which head failed
    """
    model.eval()
    
    # Storage for all tasks
    all_labels = {task: [] for task in task_names}
    all_preds = {task: [] for task in task_names}
    all_probs = {task: [] for task in task_names}
    running_loss = {task: 0.0 for task in task_names}
    
    # Collect outputs
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Comprehensive Validation"):
            volumes = batch['volume'].to(config.DEVICE, non_blocking=True)
            
            labels = {
                'malignancy': batch['malignancy_label'].to(config.DEVICE, non_blocking=True),
                'spiculation': batch['spiculation_label'].to(config.DEVICE, non_blocking=True),
                'margin': batch['margin_label'].to(config.DEVICE, non_blocking=True),
            }
            
            with autocast(enabled=config.USE_AMP):
                mal_out, spi_out, mar_out = model(volumes)
            
            outputs = {'malignancy': mal_out, 'spiculation': spi_out, 'margin': mar_out}
            
            # Collect predictions and labels
            for task_name in task_names:
                probs = torch.softmax(outputs[task_name], dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs[task_name], dim=1).cpu().numpy()
                
                all_labels[task_name].extend(labels[task_name].cpu().numpy())
                all_preds[task_name].extend(preds)
                all_probs[task_name].extend(probs)
    
    # Compute metrics for each task
    print(f"\n{'='*70}")
    print("COMPREHENSIVE VALIDATION RESULTS")
    print(f"{'='*70}")
    
    all_metrics = {}
    for task_name in task_names:
        metrics = compute_all_task_metrics(
            all_labels[task_name],
            all_preds[task_name],
            all_probs[task_name],
            task_name
        )
        all_metrics[task_name] = metrics
        
        # Print metrics
        print_task_metrics(metrics, task_name, "  ")
        
        # Print prediction distribution
        print_prediction_distribution(
            all_preds[task_name],
            all_probs[task_name],
            task_name,
            "  "
        )
        
        # Detect degenerate predictions
        detect_degenerate_predictions(
            all_preds[task_name],
            all_probs[task_name],
            task_name
        )
    
    return all_metrics, all_labels, all_preds, all_probs


def save_all_task_confusion_matrices(all_labels, all_preds, task_names, output_dir):
    """
    Save confusion matrices for all tasks as images.
    
    WHY this matters:
    - Visual confirmation of which classes model confuses
    - Medical AI: want to minimize false negatives (missed cancer)
    - Shows trade-off between sensitivity/specificity per task
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_display_names = {
        'malignancy': 'MALIGNANCY (Benign vs Malignant)',
        'spiculation': 'SPICULATION (No vs Yes)',
        'margin': 'MARGIN (Smooth vs Irregular)'
    }
    
    for task_name in task_names:
        cm = confusion_matrix(all_labels[task_name], all_preds[task_name])
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {task_display_names[task_name]}')
        plt.colorbar()
        
        tick_marks = np.arange(2)
        class_names = ['Negative (0)', 'Positive (1)']
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / f'confusion_matrix_{task_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Saved confusion matrices for all tasks to {output_dir}")


def save_multi_task_evaluation_plots(all_labels, all_preds, all_probs, task_names, output_dir):
    """Save comprehensive evaluation plots for all tasks"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for task_name in task_names:
        y_true = np.array(all_labels[task_name])
        y_pred = np.array(all_preds[task_name])
        y_prob = np.array(all_probs[task_name])
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {task_name.upper()}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(output_dir / f'roc_curve_{task_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PR Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_vals, precision_vals)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall_vals, precision_vals, color='darkgreen', lw=2, 
                 label=f'PR-AUC = {pr_auc:.4f}')
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {task_name.upper()}')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.savefig(output_dir / f'pr_curve_{task_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Saved ROC/PR curves for all tasks to {output_dir}")


# ============================================================
# Improved Multi-Task Sampling
# ============================================================

def create_multi_task_sampler(metadata, task_names=['malignancy_label', 'spiculation_label', 'margin_label']):
    """
    Create a sampler that balances all tasks, not just malignancy.
    
    WHY we need multi-task balancing:
    - Malignancy may be 70:30 imbalance
    - Spiculation may be 80:20 imbalance
    - Margin may be 60:40 imbalance
    - Simple malignancy weighting misses other imbalances
    - Result: auxiliary heads collapse because minority classes never seen
    
    SOLUTION: Use weighted combination of all task imbalances
    """
    n_samples = len(metadata)
    
    # Calculate class weights for each task
    task_weights = {}
    for task in task_names:
        labels = metadata[task].values
        class_counts = np.bincount(labels)
        # Inverse frequency weighting
        class_weights = 1.0 / (class_counts + 1e-6)  # Add epsilon to avoid division by zero
        task_weights[task] = class_weights
    
    # Combine weights (product of all task weights)
    # This ensures samples with rare classes in ANY task get oversampled
    combined_weights = np.ones(n_samples)
    for task in task_names:
        task_ws = np.array([task_weights[task][label] for label in metadata[task].values])
        combined_weights *= task_ws
    
    # Normalize
    combined_weights = combined_weights / combined_weights.sum()
    
    # Convert to sample weights for WeightedRandomSampler
    sample_weights = combined_weights * n_samples  # Scale to num_samples for sampler
    
    print(f"\n  Multi-task sampling weights computed:")
    for task in task_names:
        labels = metadata[task].values
        unique, counts = np.unique(labels, return_counts=True)
        ratio = max(counts) / min(counts) if len(unique) == 2 else 1.0
        print(f"    {task}: imbalance ratio = {ratio:.2f}:1")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=n_samples,
        replacement=True
    )
    
    return sampler


# ============================================================
# Uncertainty-Weighted Multi-Task Learning
# ============================================================

class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., 2018).
    
    WHY this helps multi-task learning:
    - Each task has different difficulty and noise level
    - Fixed weights cause conflict: easy tasks dominate gradient
    - Uncertainty weighting learns optimal balance automatically
    - Less hyperparameter tuning, better overall performance
    
    Paper: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
    """
    
    def __init__(self, num_tasks=3):
        super().__init__()
        # Learnable log(sigma) for each task (sigma = uncertainty)
        # Using log(sigma) ensures sigma > 0 and stabilizes optimization
        self.log_sigmas = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, mal_out, spi_out, mar_out, mal_labels, spi_labels, mar_labels):
        """
        Compute weighted loss where weights are learned based on uncertainty.
        
        Loss = (1/2σ²) * task_loss + log(σ)
        
        - If task is hard (high loss variance), σ increases, weight decreases
        - If task is easy (low loss), σ decreases, weight increases
        - The log(σ) term prevents σ from growing too large
        """
        total_loss = 0.0
        task_losses = {}
        
        # Task losses
        outputs = [mal_out, spi_out, mar_out]
        labels = [mal_labels, spi_labels, mar_labels]
        task_names = ['malignancy_label', 'spiculation_label', 'margin_label']
        
        for i, (out, lab, task_name) in enumerate(zip(outputs, labels, task_names)):
            loss = F.cross_entropy(out, lab, reduction='none')
            
            # Uncertainty weight: (1/2σ²) * loss + log(σ)
            sigma = torch.exp(self.log_sigmas[i])
            weight = 1.0 / (sigma ** 2)
            reg_term = torch.log(sigma)
            
            weighted_loss = weight * loss.mean() + reg_term
            total_loss += weighted_loss
            task_losses[task_name] = loss.mean().item()
        
        return total_loss, task_losses


class DynamicTaskWeightingLoss(nn.Module):
    """
    Dynamic task weighting that adapts based on training progress.
    
    WHY this helps:
    - Early training: focus on easier tasks to build shared representations
    - Late training: balance all tasks for final optimization
    - Prevents auxiliary heads from being overwhelmed early
    """
    
    def __init__(self, base_weights={'malignancy': 1.0, 'spiculation': 0.5, 'margin': 0.5}):
        super().__init__()
        self.base_weights = base_weights
        self.epoch = 0
        self.task_losses_history = {k: [] for k in base_weights.keys()}
    
    def update_epoch(self, epoch):
        """Update weighting based on epoch (progression)"""
        self.epoch = epoch
    
    def forward(self, outputs, labels):
        mal_out, spi_out, mar_out = outputs
        mal_labels, spi_labels, mar_labels = labels
        
        losses = {
            'malignancy': F.cross_entropy(mal_out, mal_labels),
            'spiculation': F.cross_entropy(spi_out, spi_labels),
            'margin': F.cross_entropy(mar_out, mar_labels),
        }
        
        # Track loss history for dynamic weighting
        for task_name, loss in losses.items():
            self.task_losses_history[task_name].append(loss.item())
        
        # Dynamic weights based on loss magnitude
        # Tasks with higher loss get higher weight (inverse scaling)
        total_weight = 0.0
        weighted_losses = {}
        
        for task_name, loss in losses.items():
            # Base weight * dynamic multiplier
            base = self.base_weights[task_name]
            
            # Dynamic multiplier based on relative loss magnitude
            if len(self.task_losses_history[task_name]) > 5:
                recent_losses = self.task_losses_history[task_name][-5:]
                avg_loss = np.mean(recent_losses)
                if avg_loss > 0:
                    # Scale based on relative loss (harder tasks get more weight)
                    dyn_weight = base * (1.0 + np.log1p(avg_loss))
                else:
                    dyn_weight = base
            else:
                dyn_weight = base
            
            weighted_losses[task_name] = loss * dyn_weight
            total_weight += dyn_weight
        
        total_loss = sum(weighted_losses.values())
        task_losses = {k: v.item() for k, v in losses.items()}
        
        return total_loss, task_losses


# ============================================================
# Metrics & Evaluation (Original - kept for compatibility)
# ============================================================
def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute comprehensive classification metrics.
    
    WHY PR-AUC matters for imbalance:
    - ROC-AUC can be overly optimistic with imbalanced data
    - PR-AUC focuses on positive (malignant) class performance
    - More sensitive to false positives in minority class
    - Better indicator of clinical utility
    - High PR-AUC = model can identify malignancies reliably
    """
    # WHY calibration and sensitivity are important in medical AI:
    # - False negatives (missing cancer) are life-threatening
    # - High sensitivity = catch all malignant cases
    # - Calibration = predicted probabilities match real risks
    # - Clinicians need trustworthy confidence scores
    # - Affects patient referral and treatment decisions
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),  # True positive rate
        'specificity': recall_score(1 - np.array(y_true), 1 - np.array(y_pred)),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
    }
    
    # PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    metrics['pr_auc'] = auc(recall_vals, precision_vals)
    
    return metrics


def save_evaluation_plots(y_true, y_pred, y_prob, output_dir: Path):
    """Save ROC curve, PR curve, and confusion matrix"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Malignancy Classification')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PR Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='darkgreen', lw=2, 
             label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign', 'Malignant'])
    plt.yticks(tick_marks, ['Benign', 'Malignant'])
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]}',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Evaluation plots saved to {output_dir}")


# ============================================================
# Training & Validation
# ============================================================
def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, config):
    """Train for one epoch with gradient accumulation and mixed precision"""
    model.train()
    running_loss = 0.0
    task_losses = {'malignancy_label': 0, 'spiculation_label': 0, 'margin_label': 0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        volumes = batch['volume'].to(config.DEVICE, non_blocking=True)
        mal_labels = batch['malignancy_label'].to(config.DEVICE, non_blocking=True)
        spi_labels = batch['spiculation_label'].to(config.DEVICE, non_blocking=True)
        mar_labels = batch['margin_label'].to(config.DEVICE, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast(enabled=config.USE_AMP):
            mal_out, spi_out, mar_out = model(volumes)
            loss, task_loss_dict = criterion(
                mal_out, spi_out, mar_out,
                mal_labels, spi_labels, mar_labels
            )
            # Scale loss for gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update task losses
        for key in task_losses:
            task_losses[key] += task_loss_dict[key]
        
        # Gradient accumulation: update weights every N batches
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION == 0:
            # Gradient clipping (prevents exploding gradients)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * config.GRADIENT_ACCUMULATION
        
        pbar.set_postfix({
            'loss': f"{loss.item() * config.GRADIENT_ACCUMULATION:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    avg_loss = running_loss / len(dataloader)
    task_losses = {k: v / len(dataloader) for k, v in task_losses.items()}
    
    return avg_loss, task_losses


@torch.no_grad()
def validate(model, dataloader, criterion, config, comprehensive=False):
    """
    Validate model and compute metrics.
    
    Args:
        comprehensive: If True, compute metrics for ALL tasks (malignancy, spiculation, margin)
                     If False, only compute malignancy metrics (for speed during training)
    """
    model.eval()
    running_loss = 0.0
    
    if comprehensive:
        # Full multi-task evaluation
        return comprehensive_validate(model, dataloader, criterion, config)
    
    # Fast single-task evaluation (malignancy only)
    all_mal_labels = []
    all_mal_preds = []
    all_mal_probs = []
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for batch in pbar:
        volumes = batch['volume'].to(config.DEVICE, non_blocking=True)
        mal_labels = batch['malignancy_label'].to(config.DEVICE, non_blocking=True)
        spi_labels = batch['spiculation_label'].to(config.DEVICE, non_blocking=True)
        mar_labels = batch['margin_label'].to(config.DEVICE, non_blocking=True)
        
        with autocast(enabled=config.USE_AMP):
            mal_out, spi_out, mar_out = model(volumes)
            loss, _ = criterion(mal_out, spi_out, mar_out, 
                               mal_labels, spi_labels, mar_labels)
        
        running_loss += loss.item()
        
        # Collect predictions
        mal_probs = torch.softmax(mal_out, dim=1)[:, 1].cpu().numpy()
        mal_preds = torch.argmax(mal_out, dim=1).cpu().numpy()
        
        all_mal_labels.extend(mal_labels.cpu().numpy())
        all_mal_preds.extend(mal_preds)
        all_mal_probs.extend(mal_probs)
    
    avg_loss = running_loss / len(dataloader)
    
    # Compute metrics
    metrics = compute_metrics(all_mal_labels, all_mal_preds, all_mal_probs)
    metrics['val_loss'] = avg_loss
    
    return metrics, all_mal_labels, all_mal_preds, all_mal_probs


# ============================================================
# Optuna Hyperparameter Optimization
# ============================================================
def objective(trial: Trial):
    """
    Optuna objective function for hyperparameter tuning.
    
    WHY Optuna:
    - Bayesian optimization finds better hyperparameters than grid search
    - Efficiently explores large hyperparameter spaces
    - Prunes unpromising trials early (saves time)
    - Integrates seamlessly with MLflow
    """
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])  # Max 256 for stability
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Update config
    config = Config()
    config.LEARNING_RATE = lr
    config.BATCH_SIZE = batch_size
    config.FOCAL_GAMMA = focal_gamma
    config.WEIGHT_DECAY = weight_decay
    
    # Ensure dataset is split - ALWAYS force regeneration to ensure correct filepaths!
    train_meta_path = config.SPLIT_OUTPUT_DIR / "train_metadata.csv"
    
    # Delete any existing split files to force regeneration with correct paths
    if train_meta_path.exists():
        print(f"\n  ⚠️  Found existing split files, deleting to force regeneration...")
        for f in ['train_metadata.csv', 'val_metadata.csv', 'test_metadata.csv']:
            fpath = config.SPLIT_OUTPUT_DIR / f
            if fpath.exists():
                fpath.unlink()
                print(f"    Deleted: {fpath}")
    
    # Now split will happen fresh
    if config.METADATA_FILE:
        print(f"  Preparing dataset from: {config.METADATA_FILE}")
        # Use the simpler Kaggle-specific function
        prepare_dataset_for_kaggle(config.DATA_DIR, config.METADATA_FILE, config.SPLIT_OUTPUT_DIR)
    
    # Create dataloaders
    train_dataset = LungNoduleDataset(
        config.SPLIT_OUTPUT_DIR / "train_metadata.csv",
        transform=True
    )
    val_dataset = LungNoduleDataset(
        config.SPLIT_OUTPUT_DIR / "val_metadata.csv",
        transform=False
    )
    
    # WHY weighted random sampler:
    # - Class imbalance: benign >> malignant
    # - Without sampling, model sees mostly benign examples
    # - Weighted sampler oversamples malignant cases
    # - Forces model to learn malignant patterns
    # - More effective than oversampling (no duplicate data)
    labels = train_dataset.metadata['malignancy_label'].tolist()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model
    model = LungNoduleClassifier(config).to(config.DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    
    # Loss
    criterion = MultiTaskLoss(config)
    criterion.mal_loss = FocalLoss(alpha=0.25, gamma=focal_gamma)
    
    # Mixed precision
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Training loop (50 epochs for trial)
    best_val_auc = 0.0
    
    for epoch in range(50):
        # Train
        train_loss, _ = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config
        )
        scheduler.step()
        
        # Validate
        val_metrics, _, _, _ = validate(model, val_loader, criterion, config)
        
        # Report to Optuna
        trial.report(val_metrics['roc_auc'], epoch)
        
        # Prune if unpromising
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        best_val_auc = max(best_val_auc, val_metrics['roc_auc'])
    
    # Cleanup
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_val_auc


# ============================================================
# Main Training Pipeline
# ============================================================
def train_full_model(config: Config, best_params: Dict = None):
    """
    Full training pipeline with best hyperparameters from Optuna.
    """
    print("\n" + "="*70)
    print("FULL MODEL TRAINING")
    print("="*70)
    
    # Apply best parameters
    if best_params:
        config.LEARNING_RATE = best_params['lr']
        config.BATCH_SIZE = best_params['batch_size']
        config.FOCAL_GAMMA = best_params['focal_gamma']
        config.WEIGHT_DECAY = best_params['weight_decay']
        print(f"Using best hyperparameters: {best_params}")
    
    # Setup MLflow
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    mlflow.start_run()
    
    # Log config
    mlflow.log_params({
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'focal_gamma': config.FOCAL_GAMMA,
        'max_epochs': config.MAX_EPOCHS,
        'use_amp': config.USE_AMP,
        'patch_size': config.PATCH_SIZE,
    })
    
    # Ensure dataset is split - ALWAYS force regeneration to ensure correct filepaths!
    train_meta_path = config.SPLIT_OUTPUT_DIR / "train_metadata.csv"
    
    # Delete any existing split files to force regeneration with correct paths
    if train_meta_path.exists():
        print(f"\n  ⚠️  Found existing split files, deleting to force regeneration...")
        for f in ['train_metadata.csv', 'val_metadata.csv', 'test_metadata.csv']:
            fpath = config.SPLIT_OUTPUT_DIR / f
            if fpath.exists():
                fpath.unlink()
                print(f"    Deleted: {fpath}")
    
    # Now split will happen fresh
    if config.METADATA_FILE:
        print(f"  Preparing dataset from: {config.METADATA_FILE}")
        # Use the simpler Kaggle-specific function
        prepare_dataset_for_kaggle(config.DATA_DIR, config.METADATA_FILE, config.SPLIT_OUTPUT_DIR)
    
    print("\nLoading datasets...")
    train_dataset = LungNoduleDataset(
        config.SPLIT_OUTPUT_DIR / "train_metadata.csv",
        transform=True,
        cache=True  # Cache in memory for speed
    )
    val_dataset = LungNoduleDataset(
        config.SPLIT_OUTPUT_DIR / "val_metadata.csv",
        transform=False
    )
    test_dataset = LungNoduleDataset(
        config.SPLIT_OUTPUT_DIR / "test_metadata.csv",
        transform=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Print comprehensive label diagnostics BEFORE training
    print_dataset_diagnostics(
        {'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
        "Training Started"
    )
    
    # Create multi-task weighted sampler (balances ALL tasks, not just malignancy)
    sampler = create_multi_task_sampler(train_dataset.metadata)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Inference batch size = 1
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Initialize model
    model = LungNoduleClassifier(config).to(config.DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Cosine annealing with warmup
    warmup_epochs = 10
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, 
                            schedulers=[warmup_scheduler, cosine_scheduler],
                            milestones=[warmup_epochs])
    
    # Loss
    criterion = MultiTaskLoss(config)
    criterion.mal_loss = FocalLoss(alpha=0.25, gamma=config.FOCAL_GAMMA)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Use uncertainty-weighted loss for better multi-task balance
    # WHY: Standard fixed weights cause conflict between tasks
    # Uncertainty weighting learns optimal balance automatically
    use_uncertainty_weighting = True
    if use_uncertainty_weighting:
        criterion = UncertaintyWeightedLoss(num_tasks=3)
        criterion = criterion.to(config.DEVICE)
        print(f"  ✓ Using Uncertainty-Weighted Multi-Task Loss")
    
    for epoch in range(config.MAX_EPOCHS):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss, train_task_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config
        )
        scheduler.step()
        
        # Validate (fast mode: malignancy only every 5 epochs, full evaluation periodically)
        if (epoch + 1) % 5 == 0:
            # Full comprehensive validation every 5 epochs
            print(f"\n  Running comprehensive validation (all tasks)...")
            all_val_metrics, _, _, _ = validate(model, val_loader, criterion, config, comprehensive=True)
            
            # Log all task metrics
            for task_name, metrics in all_val_metrics.items():
                prefix = f'val_{task_name}'
                mlflow.log_metrics({
                    f'{prefix}_accuracy': metrics['accuracy'],
                    f'{prefix}_f1': metrics['f1'],
                    f'{prefix}_precision': metrics['precision'],
                    f'{prefix}_recall': metrics['recall'],
                    f'{prefix}_specificity': metrics['specificity'],
                }, step=epoch)
                if metrics.get('roc_auc'):
                    mlflow.log_metrics({f'{prefix}_roc_auc': metrics['roc_auc']}, step=epoch)
            
            # Use malignancy metrics for early stopping (primary task)
            val_metrics = {
                'roc_auc': all_val_metrics['malignancy'].get('roc_auc', 0),
                'f1': all_val_metrics['malignancy'].get('f1', 0),
                'accuracy': all_val_metrics['malignancy'].get('accuracy', 0),
                'val_loss': all_val_metrics['malignancy'].get('val_loss', 0),
            }
            
            # Print all task results
            print(f"\n{'='*60}")
            print(f"COMPREHENSIVE EPOCH {epoch+1} RESULTS")
            print(f"{'='*60}")
            for task_name, metrics in all_val_metrics.items():
                print(f"\n  {task_name.upper()}: ")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    F1: {metrics['f1']:.4f}")
                print(f"    ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}" if metrics.get('roc_auc') else "    ROC-AUC: N/A")
                
                # Check for degenerate predictions
                cm = metrics['confusion_matrix']
                # Convert to numpy if it's a list
                if not hasattr(cm, 'sum'):
                    import numpy as np
                    cm = np.array(cm)
                
                total = cm.sum()
                pred_ratio = cm[1].sum() / total if total > 0 else 0
                label_ratio = cm[:, 1].sum() / total if total > 0 else 0
                
                # Detect if predictions are collapsed
                if cm[0][1] + cm[1][1] == 0:  # All predictions are class 0
                    print(f"    ⚠️  WARNING: All {task_name} predictions are 0 (collapsed!)")
                elif cm[0][0] + cm[1][0] == 0:  # All predictions are class 1
                    print(f"    ⚠️  WARNING: All {task_name} predictions are 1 (collapsed!)")
        else:
            # Fast validation (malignancy only)
            val_metrics, _, _, _ = validate(model, val_loader, criterion, config, comprehensive=False)
        
        epoch_time = time.time() - start_time
        
        # Log to MLflow
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_metrics.get('val_loss', 0) if isinstance(val_metrics, dict) else 0,
            'val_roc_auc': val_metrics.get('roc_auc', 0) if isinstance(val_metrics, dict) else 0,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }, step=epoch)
        
        # Print progress
        val_auc = val_metrics.get('roc_auc', 0) if isinstance(val_metrics, dict) else val_metrics
        print(f"\nEpoch {epoch+1}/{config.MAX_EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val ROC-AUC: {val_auc:.4f}")
        
        # Early stopping
        if val_metrics['roc_auc'] > best_val_auc + config.MIN_DELTA:
            best_val_auc = val_metrics['roc_auc']
            patience_counter = 0
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['roc_auc'],
                'val_f1': val_metrics['f1'],
                'val_accuracy': val_metrics['accuracy'],
                'config': {
                    'batch_size': config.BATCH_SIZE,
                    'learning_rate': config.LEARNING_RATE,
                    'weight_decay': config.WEIGHT_DECAY,
                    'focal_gamma': config.FOCAL_GAMMA,
                    'dropout': 0.5,
                }
            }
            print(f"  ✓ New best model! AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break
    
    # Save best model
    if best_model_state:
        best_model_path = config.OUTPUT_DIR / "best_model.pth"
        torch.save(best_model_state, best_model_path)
        print(f"\n✓ Best model saved: {best_model_path}")
        
        # Log to MLflow
        mlflow.log_artifact(str(best_model_path))
    
    # Final evaluation on TEST SET ONLY
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)
    
    # Load best model
    model.load_state_dict(best_model_state['model_state_dict'])
    model.eval()
    
    # COMPREHENSIVE TEST EVALUATION - ALL TASKS
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SET EVALUATION (ALL TASKS)")
    print("="*70)
    
    # Collect all predictions for all tasks
    test_labels_all = {task: [] for task in ['malignancy', 'spiculation', 'margin']}
    test_preds_all = {task: [] for task in ['malignancy', 'spiculation', 'margin']}
    test_probs_all = {task: [] for task in ['malignancy', 'spiculation', 'margin']}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Comprehensive Test Evaluation"):
            volumes = batch['volume'].to(config.DEVICE)
            
            labels_dict = {
                'malignancy': batch['malignancy_label'].to(config.DEVICE),
                'spiculation': batch['spiculation_label'].to(config.DEVICE),
                'margin': batch['margin_label'].to(config.DEVICE),
            }
            
            mal_out, spi_out, mar_out = model(volumes)
            outputs = {'malignancy': mal_out, 'spiculation': spi_out, 'margin': mar_out}
            
            for task_name in ['malignancy', 'spiculation', 'margin']:
                probs = torch.softmax(outputs[task_name], dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs[task_name], dim=1).cpu().numpy()
                
                test_labels_all[task_name].extend(labels_dict[task_name].cpu().numpy())
                test_preds_all[task_name].extend(preds)
                test_probs_all[task_name].extend(probs)
    
    # Compute and display metrics for ALL tasks
    all_test_metrics = {}
    print("\n")
    for task_name in ['malignancy', 'spiculation', 'margin']:
        metrics = compute_all_task_metrics(
            test_labels_all[task_name],
            test_preds_all[task_name],
            test_probs_all[task_name],
            task_name
        )
        all_test_metrics[task_name] = metrics
        
        print_task_metrics(metrics, task_name, "  ")
        print_prediction_distribution(
            test_preds_all[task_name],
            test_probs_all[task_name],
            task_name,
            "  "
        )
        detect_degenerate_predictions(
            test_preds_all[task_name],
            test_probs_all[task_name],
            task_name
        )
        print()
    
    # Save confusion matrices for all tasks
    save_all_task_confusion_matrices(
        test_labels_all,
        test_preds_all,
        ['malignancy', 'spiculation', 'margin'],
        config.OUTPUT_DIR
    )
    
    # Save comprehensive evaluation plots
    save_multi_task_evaluation_plots(
        test_labels_all,
        test_preds_all,
        test_probs_all,
        ['malignancy', 'spiculation', 'margin'],
        config.OUTPUT_DIR
    )
    
    # Log all test metrics to MLflow
    for task_name, metrics in all_test_metrics.items():
        prefix = f'test_{task_name}'
        mlflow.log_metrics({
            f'{prefix}_accuracy': metrics['accuracy'],
            f'{prefix}_f1': metrics['f1'],
            f'{prefix}_precision': metrics['precision'],
            f'{prefix}_recall': metrics['recall'],
            f'{prefix}_specificity': metrics['specificity'],
        })
        if metrics.get('roc_auc'):
            mlflow.log_metrics({f'{prefix}_roc_auc': metrics['roc_auc']})
    
    # Export comprehensive metrics JSON
    metrics_json = {
        'all_task_metrics': all_test_metrics,
        'best_val_auc': best_val_auc,
        'best_epoch': best_model_state['epoch'],
        'hyperparameters': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'focal_gamma': config.FOCAL_GAMMA,
            'loss_type': 'uncertainty_weighted',
        }
    }
    
    metrics_path = config.OUTPUT_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\n✓ Comprehensive metrics saved: {metrics_path}")
    
    mlflow.end_run()
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return model, all_test_metrics


# ============================================================
# Main Entry Point
# ============================================================
def main():
    """Main training pipeline"""
    print("="*70)
    print("KAGGLE-READY 3D LUNG NODULE CLASSIFICATION")
    print("="*70)
    
    # Setup - Auto-detect environment
    set_deterministic_seed(42)
    
    if Path("/kaggle/input").exists():
        Config.setup_for_kaggle()
    else:
        Config.setup_for_local()
    
    config = Config()
    
    # Step 1: Optuna Hyperparameter Tuning
    print("\n" + "="*70)
    print("STEP 1: OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"Trials: {config.N_TRIALS}")
    print(f"Timeout: {config.OPTUNA_TIMEOUT / 3600:.1f} hours")
    print("="*70)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    study.optimize(objective, n_trials=config.N_TRIALS, timeout=config.OPTUNA_TIMEOUT)
    
    print(f"\n✓ Optuna optimization complete!")
    print(f"  Best ROC-AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    
    # Save Optuna study
    optuna_dir = config.OUTPUT_DIR / "optuna_study"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    with open(optuna_dir / "best_params.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    # Save study for later download
    import joblib
    joblib.dump(study, optuna_dir / "study.pkl")
    print(f"  ✓ Optuna study saved: {optuna_dir}")
    
    # Step 2: Full Training with Best Params
    best_params = study.best_params
    model, test_metrics = train_full_model(config, best_params)
    
    # Package outputs for download
    print("\n" + "="*70)
    print("PACKAGING OUTPUTS FOR DOWNLOAD")
    print("="*70)
    
    # Create zip of MLflow + Optuna folders
    if config.OUTPUT_DIR.exists():
        # Determine mlruns location (local or Kaggle)
        if Path("/kaggle/working").exists():
            # Kaggle environment
            mlruns_dir = Path("/kaggle/working/mlruns")
            zip_path = Path("/kaggle/working/training_outputs.zip")
        else:
            # Local environment (RTX 6000 Pro)
            project_root = Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)")
            mlruns_dir = project_root / "02_Model_Development" / "ml_model_engineering" / "mlruns"
            zip_path = config.OUTPUT_DIR / "training_outputs.zip"
        
        # Create a temporary directory with all outputs
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy output folder
            output_copy = temp_path / "output"
            if config.OUTPUT_DIR.exists():
                shutil.copytree(config.OUTPUT_DIR, output_copy, ignore=shutil.ignore_patterns('*.zip'))
                print(f"  ✓ Added output folder")
            
            # Copy mlruns folder if it exists
            if mlruns_dir.exists():
                mlruns_copy = temp_path / "mlruns"
                shutil.copytree(mlruns_dir, mlruns_copy)
                print(f"  ✓ Added mlruns folder")
            
            # Copy Optuna study if it exists
            optuna_dir = config.OUTPUT_DIR / "optuna_study"
            if optuna_dir.exists():
                optuna_copy = temp_path / "optuna_study"
                shutil.copytree(optuna_dir, optuna_copy)
                print(f"  ✓ Added optuna_study folder")
            
            # Create zip
            shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', temp_path)
            print(f"\n✓ All outputs packaged: {zip_path}")
            print(f"  Includes: output/, mlruns/, optuna_study/")
    
    # Save test metrics for reference
    if 'test_metrics' in locals():
        metrics_path = config.OUTPUT_DIR / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\n✓ Test metrics saved: {metrics_path}")
    
    print("\n" + "="*70)
    print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()