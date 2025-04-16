# 🧭 Workflow: Two-Stage Dengue Breeding Site Detection System

This document outlines the end-to-end process for running the dengue breeding site detection pipeline using deep learning models and remote sensing data.

---

## 🧩 Step 1: Slicing Orthophotos

- **Locations:** Mirpur Technical and Kamalapur.
- **Tool:** Kaggle notebooks (soon to be ported to Google Colab).
- **Output:** Image slices prepared for training and inference.
(NB): For Object Detection and Building Segmentation : Refer to Kaggle Notebook (denguebsi)
---

## 🧠 Step 2: Model Training & Inference

- **Training:** Done using Mirpur Technical slices (For both OD and BS) : Refer to Roboflow (For both OD and BS)
- **Inference Target:** Kamalapur (OD (326 x512 tiles), BD (with two slicing variants))
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
(NB): bbox_per_image.py

### ➤ YOLO Variants:
- **Type:** Instance segmentation (via Ultralytics).
- **Output:**
  - Bounding box + segmented polygon (corner points).
  - Confidence score.
- **Use:** Outputs feed into sliding window technique for post-processing.
- **Note:** Polygon extraction and mask generation done in a Colab notebook using `lib` object from Ultralytics inference.
(NB): YOLOv9-GELAN-C (OD): [YOLO_OBJECT_DETECTION](https://colab.research.google.com/drive/1IN_Ejbd9m13jd5o2u4fgu6sk4hM28CLr?usp=sharing), YOLO Variants (BS): Refer to colab notebook [YOLO_INSTANCE_SEGMENTATION](https://colab.research.google.com/drive/18xOC6Wvi0VgLvvKuHNfvNJxXA3PEf3V5?usp=sharing)
---

More steps will be added as you provide them. Ready when you are!
