# 🧭 Workflow: Two-Stage Dengue Breeding Site Detection System

This document outlines the end-to-end process for running the dengue breeding site detection pipeline using deep learning models and remote sensing data.

---

## 📂 Dataset (Annotated)

- **Object Detection:**  
  [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/thesis19/mv_dataset)

- **Building Segmentation:**
  - **Mirpur Technical (Train-set):**  
    [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/thesis19/mt-seg-ann-1024_compressed_cv2-lyqkb)
  - **Kamalapur (Test-set):**
    - `52 images`  
      [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/thesis19/kpur-seg-ann-1024-compressed-cv2-jnwtg)
    - `Full orthophoto (289 buildings)`  
      [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/thesis19/buildingsegmentation1-wdugm)

---

## 🧩 Step 1: Slicing Orthophotos

- **Locations:** Mirpur Technical and Kamalapur  
- **Tool:** Kaggle notebooks (soon to be ported to Google Colab)  
- **Output:** Image slices prepared for training and inference  
- **Notebook References:**
  - **Dataset Slicing + Preparation:**  
    Kaggle Notebook: `denguebsi` (Object Detection + Building Segmentation slicing)
---

## 🧠 Step 2: Model Training & Inference

- **Training:**  
  Using Mirpur Technical slices for both Object Detection and Building Segmentation  
- **Inference Target:** Kamalapur  
  - `52 slices` – Testing segmentation accuracy  
  - `200 slices` – Sliding window-based OBB generation  
    - Overlapped (50%) vs Non-overlapped variants  
  - Inference on 326x512 tiles (OD) and full orthophoto (BD)

- **Notebook References:**
  - **YOLOv9-GELAN-C (Object Detection):**  
    [YOLO_OBJECT_DETECTION Colab](https://colab.research.google.com/drive/1IN_Ejbd9m13jd5o2u4fgu6sk4hM28CLr?usp=sharing)

  - **YOLO Variants (Instance Segmentation - BS):**  
    [YOLO_INSTANCE_SEGMENTATION Colab](https://colab.research.google.com/drive/18xOC6Wvi0VgLvvKuHNfvNJxXA3PEf3V5?usp=sharing)

---

## 🧪 Step 3: Model Evaluation

### ➤ SegGPT
- **Type:** In-context learning  
- **Output:** Highlighted pixel regions representing segmented buildings  
- **Script Used:** `bbox_per_image.py`

### ➤ YOLO Variants
- **Type:** Instance segmentation via Ultralytics  
- **Output:** Bounding box + segmented polygon + confidence score  
- **Use:** Outputs processed for post-inference sliding window technique  
- **Tools:** `lib` object from Ultralytics used for polygon extraction and mask generation  

> 🔍 **Note:** Segmentation model evaluation is computed using the `seg_accuracy.m` (along with some additional scripts) MATLAB script.
