# MATLAB Retinal Image Analysis Add-on (RetinaSense AI)

This directory provides an optional, independent image-processing companion to **RetinaSense AI**.
Its calculations and outputs are strictly for quality estimation and visualization enhancement; they are **never** fed into the EfficientNetB3 disease prediction or Grad-CAM pipeline.

---

## 1. System Requirements

* **MATLAB**: R2022b, R2023a, R2023b, R2024a, or R2024b with **Image Processing Toolbox**.
* **Python Environment**: A Python version supported by your specific MATLAB release:
  * MATLAB R2024b: Python 3.10, 3.11, 3.12
  * MATLAB R2024a: Python 3.9, 3.10, 3.11, 3.12
  * MATLAB R2023b: Python 3.9, 3.10, 3.11
  * MATLAB R2023a: Python 3.8, 3.9, 3.10
  *(Note: Official MATLAB Engine support for Python 3.13 requires MATLAB R2024b with the latest engine update; verify via `pyenv` in MATLAB).*
* **MATLAB Engine API for Python**: Installed from the local MATLAB installation into the FastAPI backend's Python environment.

---

## 2. Installation & Setup

### Step 1: Verify MATLAB & Toolbox
Open MATLAB on the native host and execute:
```matlab
ver
pyenv
```
Confirm:
1. `Image Processing Toolbox` appears in the product list.
2. `pyenv` points to or is compatible with the Python environment used by the backend.

### Step 2: Install MATLAB Engine API for Python
Navigate to the MATLAB Python engine folder and install it into your active Python environment:

**Windows (PowerShell as Administrator or within virtualenv):**
```powershell
# Example path for MATLAB R2024a:
cd "C:\Program Files\MATLAB\R2024a\extern\engines\python"
python -m pip install .
```

**Linux / macOS:**
```bash
cd "/usr/local/MATLAB/R2024a/extern/engines/python"
python -m pip install .
```

### Step 3: Verify Python Bridge
Verify that the engine imports correctly in your backend Python environment:
```bash
python -c "import matlab.engine; print('MATLAB Engine OK')"
```

---

## 3. MATLAB Functions

All functions in this folder operate deterministically and output JSON strings:

### A. `fundus_quality.m`
* **Purpose**: Measures retinal fundus image quality using deterministic engineering metrics (0–100 scale).
* **Metrics**:
  * `brightness_score`: Penalizes under-exposed (<0.35) and over-exposed (>0.75) mean intensities.
  * `contrast_score`: Evaluates standard deviation of intensity across valid tissue (target >= 0.12).
  * `sharpness_score`: Evaluates variance of the discrete 3x3 Laplacian operator over tissue.
  * `fov_score`: Measures coverage of the connected retinal field relative to frame size.
  * `quality_score`: Weighted composite score (`0.25*brightness + 0.25*contrast + 0.30*sharpness + 0.20*fov`).
  * `status`: Categorized into `'GOOD'` (>=75), `'ACCEPTABLE'` (>=50), or `'POOR'` (<50).
* **Usage**:
  ```matlab
  res = fundus_quality('test_fundus.png');
  ```

### B. `enhance_fundus.m`
* **Purpose**: Generates an enhanced fundus image for clinical visualization without modifying or overwriting the original image.
* **Pipeline**:
  1. Guard against overwriting: errors if `outputPath` equals `imagePath`.
  2. Green channel extraction (highest contrast for retinal vasculature).
  3. Slow illumination correction via large-scale Gaussian filtering.
  4. Contrast-Limited Adaptive Histogram Equalization (CLAHE).
  5. Mild Gaussian denoising and subtle unsharp masking.
  6. Recombination with color context in HSV color space.
  7. Writes to the new `outputPath` file.
* **Usage**:
  ```matlab
  enhance_fundus('original.png', 'enhanced_output.png');
  ```

### C. `retinal_features.m`
* **Purpose**: Extracts non-diagnostic quantitative image-processing descriptors.
* **Metrics**:
  * `mean_intensity`: Average green-channel intensity in retinal field.
  * `contrast`: Standard deviation of pixel values within retinal field.
  * `bright_region_percentage`: Percentage of field with pixel intensity > 0.80.
  * `dark_region_percentage`: Percentage of field with pixel intensity < 0.15.
  * `retinal_field_area`: Percentage of the frame occupied by non-background retinal field.
  * `vessel_density`: Estimated vessel-like structure area percentage using background subtraction and thresholding.
* **Usage**:
  ```matlab
  features = retinal_features('test_fundus.png');
  ```

---

## 4. Architecture & Safety Rules

```text
       Upload Scan
            │
            ▼
   Fundus Validation
            │
            ├─────────────────────────────────────────┐
            │ (Original Image)                        │ (Original Image Reference)
            ▼                                         ▼
   Existing Preprocessing                      MATLAB Add-On
            │                                         │
            ▼                                         ├─ Quality Analysis (0-100)
     EfficientNetB3                                   ├─ Enhancement (separate PNG)
            │                                         └─ Quantitative Measurements
            ▼                                         │
    Disease Prediction                                ▼
     & Grad-CAM Heatmap                     Returned to Frontend UI
            │                                 (Independent Card)
            ▼
    Clinical Report (PDF)
```

1. **Complete Independence**: The MATLAB engine is initialized **lazily** on the first `/api/matlab/analyze` call, not at server startup.
2. **Graceful Fallback**: If MATLAB or `matlab.engine` is not installed or unavailable, the endpoint returns:
   ```json
   {
     "matlab_available": false,
     "matlab_status": "MATLAB analysis unavailable"
   }
   ```
   The primary diagnosis, Grad-CAM, and report generation continue working with zero disruption.
3. **Non-Clinical Disclosure**: All quality scores and quantitative measurements are image-processing measurements, not medical diagnoses.
4. **File Safety**: Original image files are never altered or overwritten. Enhanced images are saved with unique UUID-tagged filenames in the uploads directory.
