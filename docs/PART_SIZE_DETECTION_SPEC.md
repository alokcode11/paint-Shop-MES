# Technical Specification & Requirement Document: Part Size & Variant Detection

**Project:** Industrial Paint Shop MES (AC ODU Component Verification)  
**Plant:** NGM PG Supa  
**Module:** Automated Size / Dimensional Variant Classification  

---

## 1. Executive Summary & Problem Definition
In the AC Outdoor Unit (ODU) paint shop conveyor line, identical component categories (e.g., `front_panel`, `base_panel`) arrive in different physical sizes (e.g., **20-inch**, **24-inch**, **28-inch**). 

The MES vision system must not only classify the component type but also reliably classify its **dimensional size/variant** for accurate inventory tracking, production tally, and recipe verification.

---

## 2. Technical Feasibility & Constraints

### Fundamental Computer Vision Law:
A standard 2D camera has no native depth perception. An object's pixel dimensions change depending on distance from lens.  
Therefore, dimensional measurement from 2D frames is valid **only under strict physical constraints**.

### Mandatory Factory Constraints:
1. **Fixed Camera Mounting:** Camera must be rigid, vibration-free, and mounted perpendicular (90° normal) to the conveyor travel plane.
2. **Fixed Object-to-Lens Distance (Z-Axis Constant):** Overhead conveyor rail maintains a fixed distance (e.g., 1.2 m ± 20 mm) between parts and the camera lens.
3. **No Parallax / Severe Tilt:** Parts must hang naturally without excessive swinging (< ±5°).

---

## 3. Technology Stack & Required Python Libraries

| Library | Version | Purpose in Size Detection |
| :--- | :--- | :--- |
| **`ultralytics` (YOLOv8)** | `>= 8.0.0` | Detects the component, outputs normalized/absolute bounding box coordinates `[x1, y1, x2, y2]`. |
| **`opencv-python` (cv2)** | `>= 4.8.0` | Computes pixel contours, minimum area rotated bounding rects (`cv2.minAreaRect`), aspect ratio, and perspective transformation. |
| **`numpy`** | `>= 1.24.0` | High-speed vector arithmetic for calculating Euclidean distances, bounding box diagonals, and ratios. |
| **`scipy`** (Optional) | `>= 1.10.0` | Spatial distance calculations between reference marker points (`scipy.spatial.distance`). |

---

## 4. Engineering Approaches for Size Detection

### Approach A: Pixel-to-Metric Calibration (Recommended for Static Distances)
Calibrate the physical mm-per-pixel ratio on the conveyor plane using a known physical reference object (e.g., standard 20" panel = 500 mm width).

- **Formula:**
  $$\text{PPM (Pixels Per Millimeter)} = \frac{\text{Measured Pixel Dimension}}{\text{Actual Known Metric Dimension (mm)}}$$
  $$\text{Estimated Metric Dimension (mm)} = \frac{\text{Detected Bounding Box Pixels}}{\text{PPM}}$$

```python
# Sizing Logic Example:
def classify_panel_size(pixel_width, ppm=2.45):
    real_width_mm = pixel_width / ppm
    if 470 <= real_width_mm <= 530:
        return "20_INCH"
    elif 570 <= real_width_mm <= 630:
        return "24_INCH"
    else:
        return "UNKNOWN_VARIANT"
```

---

### Approach B: Aspect Ratio & Geometric Feature Ratios (Rotation-Invariant)
When parts hang tilted/diagonal (as seen in factory frames):
- Compute the Rotated Minimum Area Rectangle: `cv2.minAreaRect(contour)`
- Extract invariant aspect ratio: $\text{Aspect Ratio} = \frac{\text{Length}}{\text{Width}}$
- Different ODU sizes usually have distinct length-to-width proportions and cutout locations.

---

### Approach C: Multi-Class Direct AI Training (Highest Accuracy)
Instead of a single class `front_panel`, train the model with variant-specific classes:
- `front_panel_20in`
- `front_panel_24in`

The YOLO neural network automatically learns subtle visual cues (stamping ribs, screw holes, internal fan shroud diameter) without needing manual pixel math.

---

## 5. Short Notes & Takeaways

- **Is a dedicated 1-line Python library available?** No. 2D camera geometry requires OpenCV geometric calibration or AI class splitting.
- **Can Paint Shop MES do this?** Yes, because overhead conveyor rails have a fixed, repeatable distance to the camera lens.
- **Best Practical Route:**
  1. Use **OpenCV bounding dimension thresholds** for immediate size filtering.
  2. For long-term 99.9% reliability, label separate classes (`front_panel_20`, `front_panel_24`) in **AnyLabeling**.

