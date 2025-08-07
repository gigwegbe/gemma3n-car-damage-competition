# AutoVisionInspector Web App – Mockup Description

**AutoVisionInspector** is a web-based application that functions as an **Evaluator or Judge** for assessing car damage detection models. It provides a user-friendly interface for uploading images, selecting models, and visualizing detection results with bounding boxes and evaluation metrics.

---

## Application Flow

### 1. User Input Stage

- **Image Upload/Selection**
  - Users can upload or select an image of a damaged vehicle.

- **Model Selection**
  - A dropdown or selection tool allows users to choose a machine learning model for evaluation.

- **Bounding Box Input**
  - Ground-truth bounding boxes may be uploaded or generated for comparison with the model output.

---

### 2. Processing Stage

- **Inference Time**
  - A delay of approximately `20ms` is noted, indicating fast processing speed for model evaluation.

- **Damage Tagging**
  - Detected damages are tagged or categorized using labels such as `crack`.

---

### 3. Output Stage

- **Image Display**
  - The selected image is shown in the output panel.

- **Bounding Box Overlay**
  - Bounding boxes are drawn over detected damage regions.
  - Each box may include severity labels or confidence scores (e.g., High / Medium / Low).

---

## Interface Components

- **Image Upload Panel**
- **Model Selection Dropdown**
- **Bounding Box Input Field or Upload**
- **Output Viewer**
  - Displays image, bounding boxes, labels, and evaluation results.

---

## Purpose

AutoVisionInspector allows users to:

- Compare model predictions with ground-truth annotations
- Visually inspect and validate damage detection performance
- Use it as part of a benchmarking or competition platform for model evaluation

---
