# 🦟 Two-Stage Dengue Breeding Site Detection System

This project presents a two-stage deep learning-based approach to identify **risky buildings** that may contribute to dengue outbreaks, using aerial imagery and object detection techniques.

---

## 🚀 Stage 1: Building Detection

- **Objective:** Identify individual buildings from orthophoto images.
- **Method:**
  - Used a custom object detection model (e.g., YOLO) on sliced orthophotos.
  - Handled overlapping slices and multiple bounding boxes per building.
  - Bounding box format: oriented (8-point with confidence score).
- **Output:** Coordinates of each detected building.

---

## 🧪 Stage 2: Dengue Breeding Factor Detection

- **Objective:** Detect objects related to dengue breeding inside/around buildings.
- **Detected Factors:**
  - Flower pots
  - Open tanks
  - Tyres
  - Polythene/plastic containers
  - Construction sites
  - Water reservoirs
- **Method:**
  - Applied object detection models within each building's bounding box.
  - Classified buildings as **risky** if any breeding factor is found within.

---

## 📈 Performance

- Achieved **\~90% detection accuracy and \~84% Balanced Accuracy while saving 35% inspection costs.**
- Ground truth validated with manual labeling and hyperparameter optimization.

---

## 🎯 Outcome

- A scalable method for early dengue risk mapping using remote sensing and AI.
- Can assist city authorities in **targeted mosquito control and awareness campaigns**.



# Directory Configuration required for the program paths

```
.
├── droneImages
├── illustrations
├── model
│   └── __pycache__
├── output
│   ├── missed_buildings
│   ├── One-Stage
│   └── YOLO-v8m
├── resources
│   ├── Building
│   │   ├── bbox_perslice
│   │   │   ├── SegGPT
│   │   │   ├── YOLO-v11m
│   │   │   └── YOLO-v8m
│   │   ├── masks
│   │   │   ├── 200_masks_seggpt
│   │   │   ├── 52_images_seggpt
│   │   │   ├── 52_masks_v11m
│   │   │   ├── 52_masks_v8m
│   │   │   ├── gt_masks
│   │   │   ├── labels_11m
│   │   │   └── labels_8m
│   │   └── test
│   ├── ClassifierOutput
│   │   ├── Building Classification
│   │   │   ├── Ground Truth
│   │   │   ├── Ground-Truth-Buildings
│   │   │   ├── SegGPT
│   │   │   ├── YOLO-v11m
│   │   │   ├── YOLO-v11m-nms
│   │   │   ├── YOLO-v11m ONE-STAGE
│   │   │   ├── YOLO-v8m
│   │   │   └── YOLO-v8m-nms
│   │   └── Object Detection
│   │       ├── Ground Truth
│   │       └── YOLO-v9c
│   ├── Objects
│   │   ├── detect
│   │   │   ├── exp
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── exp2
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   └── exp3
│   │   │       ├── images
│   │   │       └── labels
│   │   ├── detect_yolov11m
│   │   └── gt_yolov9
│   │       ├── images
│   │       └── labels
│   └── Orthophoto
└── utils
    └── __pycache__

55 directories
```
