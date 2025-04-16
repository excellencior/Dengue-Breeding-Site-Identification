# 🧭 Workflow: Two-Stage Dengue Breeding Site Detection System

This document outlines the end-to-end process for running the dengue breeding site detection pipeline using deep learning models and remote sensing data.

---

## 🧩 Step 1: Slicing Orthophotos

- **Locations:** Mirpur Technical and Kamalapur.
- **Tool:** Kaggle notebooks (soon to be ported to Google Colab).
- **Output:** Image slices prepared for training and inference.

---

## 🧠 Step 2: Model Training & Inference

- **Training:** Done using Mirpur Technical slices.
- **Inference Target:** Kamalapur, with two slicing variants:
  - **Overlapped slices** (50% in all directions).
  - **Non-overlapped slices**.

- From Kamalapur orthophoto:
  - `52 slices`: Used for **testing segmentation accuracy**.
  - `200 slices`: Used in **sliding window-based Oriented Bounding Box (OBB)** generation.

---

## 🧪 Step 3: Model Evaluation

### ➤ SegGPT:
- **Type:** In-context learning.
- **Output:** Highlighted pixel regions representing segmented buildings.

### ➤ YOLO Variants:
- **Type:** Instance segmentation (via Ultralytics).
- **Output:**
  - Bounding box + segmented polygon (corner points).
  - Confidence score.
- **Use:** Outputs feed into sliding window technique for post-processing.
- **Note:** Polygon extraction and mask generation done in a Colab notebook using `lib` object from Ultralytics inference.

---

More steps will be added as you provide them. Ready when you are!
