# 🦟 Two-Stage Dengue Breeding Site Detection System

This project presents a two-stage deep learning-based approach to identify **risky buildings** that may contribute to dengue outbreaks, using aerial imagery and object detection techniques.

---

## 🚀 Stage 1: Object Detection and Building Segmentation

- **Objective:** Detect dengue breeding factors and segment nearby buildings from orthophoto images.
- **Method:**
  - Used object detection models to locate breeding factors such as flower pots, open tanks, tyres, etc.
  - Simultaneously performed building segmentation using models like SegGPT or YOLO variants.
  - Generated bounding boxes or polygon masks for buildings and breeding factors.
- **Detected Factors:**
  - Flower pots
  - Open tanks
  - Tyres
  - Polythene/plastic containers
  - Construction sites
  - Water reservoirs
- **Output:** Localized breeding factors and segmented building structures with spatial coordinates.

---

## 🧪 Stage 2: Building Classification for Scalable Inspections

- **Objective:** Classify buildings as risky or non-risky based on proximity to detected breeding factors.
- **Method:**
  - Calculated spatial distance between each segmented building and detected breeding factors.
  - Applied rules or threshold distances to determine building risk status.
  - Used confidence scores from models to weigh reliability of predictions.
- **Output:** Risk-labeled buildings for targeted intervention.

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
