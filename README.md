This module provides a robust, PyTorch-native data ingestion and preprocessing pipeline engineered specifically for semantic segmentation tasks involving complex, multi-format aerial imagery.

normalize_img(arr): Dynamically scales multidimensional arrays to a strict [0,1] range, ensuring mathematical stability during gradient descent. It intelligently manages non-standard bit-depths (such as 11-bit or 12-bit sensor data stored as uint16) frequently encountered in satellite imagery, effectively preventing overflow errors prior to tensor conversion.

load_image_any(path, target_size, bands): A unified ingestion utility that processes standard RGB formats via PIL and complex multispectral GeoTIFFs via rasterio. It systematically extracts specific spectral bands and dynamically pads missing channels to guarantee a standardized (C, H, W) tensor shape. This ensures seamless compatibility with pre-trained image encoders commonly used in UNet architectures.

load_mask_png(path, target_size): Ingests and transforms grayscale ground-truth masks into (1,1,H,W) PyTorch tensors. It strictly enforces nearest-neighbor interpolation during spatial resizing to prevent the creation of fractional pixel values, guaranteeing that categorical class boundaries remain discrete for accurate evaluation against loss functions like BCEWithLogitsLoss.

show_image_and_mask(img, mask, alpha): Renders alpha-blended overlays combining spatial image data with binary ground-truth masks. This visualization utility is critical for rapid sanity checking, enabling the direct debugging of spatial alignments, pixel-wise loss maps, and feature activation channels prior to initiating a training loop.

Technical Expertise & Impact Summary
This workflow demonstrates a highly adaptable data engineering approach, effectively resolving common bottlenecks encountered when preparing raw remote sensing data for deep learning applications.

Advanced Tensor Architecture: Engineered dynamic data-type normalization and robust tensor permutations (translating [H, W, C] to [C, H, W]). This guarantees high-fidelity, mathematically stable transformations optimized directly for PyTorch computer vision workflows.

Geospatial & AI Convergence: Successfully bridged traditional geospatial analysis with modern deep learning by fusing rasterio and PyTorch. Gracefully handled spatial edge cases—such as missing TIF bands—by replicating available channels, satisfying the structural requirements of standard convolutional neural networks.

Loss Optimization Readiness: Structured the mask ingestion protocol specifically to preserve discrete class labels during spatial transformations. By preventing interpolation artifacts from corrupting ground-truth data, this approach directly supports stable backpropagation in spatial loss mapping.

Production-Grade Engineering: Designed a modular, type-safe data layer that abstracts the complexities of mixed-format file reading. This architecture significantly accelerates the iterative debugging cycle for model state dictionaries and layer activations by guaranteeing reliable, standardized tensor inputs.

debug.ipynb

# Aerial Imagery Semantic Segmentation Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A robust data ingestion, normalization, and preprocessing pipeline built with PyTorch and Rasterio, optimized for high-resolution aerial imagery and computer vision semantic segmentation tasks.

## Key Features

* **Multi-Format Ingestion:** Seamlessly ingests standard image formats (PNG, JPG) alongside multi-band geospatial TIFFs via `rasterio`.
* **Dynamic Bit-Depth Normalization:** Automatically scales varying bit-depths (`uint8`, `uint16`) to a normalized `[0, 1]` tensor range with built-in zero-division protection.
* **Intelligent Resizing:** Applies bilinear interpolation for continuous image features and nearest-neighbor interpolation for discrete segmentation masks to protect sharp class boundaries.
* **Visual Verification Utilities:** Includes built-in plotting functions for generating alpha-blended spatial mask overlays to verify ground-truth alignment prior to model ingestion.

---

## Technical Specifications

| Component | Input Format | Output Tensor Shape | Interpolation Method | Data Type |
| :--- | :--- | :--- | :--- | :--- |
| **`load_image_any`** | TIFF, PNG, JPG | `(1, 3, H, W)` | Bilinear | `torch.float32` |
| **`load_mask_png`** | PNG (Grayscale) | `(1, 1, H, W)` | Nearest Neighbor | `torch.float32` |
| **`normalize_img`** | Raw Array / Tensor | `(C, H, W)` | Min-Max Scaling | `torch.float32` |

---

## Requirements

* Python 3.8 or higher
* PyTorch
* Rasterio
* NumPy
* Matplotlib
* Pillow

---

## Quick Start Guide

```python
import torch
from segmentation_pipeline import load_image_any, load_mask_png, show_image_and_mask

# Define target dimensions
target_size = (512, 512)

# Load and normalize an aerial TIFF image
image_tensor = load_image_any("data/raw/aerial_tile.tif", target_size=target_size)

# Load and resize the corresponding ground-truth mask
mask_tensor = load_mask_png("data/masks/aerial_tile_mask.png", target_size=target_size)

# Verify spatial alignment visually
show_image_and_mask(image_tensor, mask_tensor)


├── segmentation_pipeline.py    # Core ingestion and preprocessing functions
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
