# Detection of Dengue Breeding Sites in Large‑scale Landscapes Plagued by Unplanned Urban Development from Aerial Imagery and Remote Sensing with Deep Learning

Early detection of dengue breeding sites is crucial for controlling outbreaks in dense urban regions where manual inspections aren't scalable or cost‑effective. This repository implements a novel two‑stage deep learning pipeline on aerial imagery and remote sensing data to - \
(1) segment individual buildings, \
(2) detect potential dengue breeding objects, and \
(3) spatially combine the two for geo‑targeted inspection. \
In orthophotos from two dengue‑prone districts in Dhaka, our two‑stage method achieved an **83.6 % balanced accuracy** with a **35 % reduction** in manual inspection cost, outperforming a one‑stage baseline (69.5 % balanced accuracy).

<p align="center">
  <img src="Illustrations%20%5Bshowcase%5D/Simple_Methodology_Diagram.png" alt="Methodology Diagram" width="600"/>
</p>


## 📂 Repository Structure

```
.
├── data [train, test]
│   ├── seggpt
│   │   ├── gt_masks
│   │   ├── inference
│   │   └── seggpt
│   ├── yolov11m
│   │   └── inference
│   └── yolov8m
│       └── inference
├── Illustrations [showcase]
├── model
│   └── __pycache__
├── notebooks
├── output
│   └── YOLO-v8m
├── resources
│   ├── Building
│   │   ├── bbox_perslice
│   │   │   ├── SegGPT
│   │   │   ├── YOLO-v11m
│   │   │   ├── YOLO-v11m-25p
│   │   │   ├── YOLO-v11m-75p
│   │   │   ├── YOLO-v8m
│   │   │   ├── YOLO-v8m-25p
│   │   │   └── YOLO-v8m-75p
│   │   ├── masks
│   │   │   ├── 52_images_seggpt
│   │   │   ├── 52_masks_v11m
│   │   │   ├── 52_masks_v8m
│   │   │   ├── labels_11m
│   │   │   ├── labels_11m_25
│   │   │   ├── labels_11m_75
│   │   │   ├── labels_8m
│   │   │   ├── labels_8m_25
│   │   │   └── labels_8m_75
│   │   └── test
│   ├── ClassifierOutput
│   │   ├── Building Classification
│   │   │   ├── Ground Truth
│   │   │   ├── Ground-Truth-Buildings
│   │   │   ├── SegGPT
│   │   │   ├── YOLO-v11m
│   │   │   ├── YOLO-v11m-25p
│   │   │   ├── YOLO-v11m-75p
│   │   │   ├── YOLO-v11m-nms
│   │   │   ├── YOLO-v11m ONE-STAGE
│   │   │   ├── YOLO-v8m
│   │   │   ├── YOLO-v8m-25p
│   │   │   ├── YOLO-v8m-75p
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
├── utils
│   └── __pycache__
└── weights
    ├── yolov11m
    └── yolov8m
```


## 🔗 Data & Pre-trained Weights

### Test Sets (orthophoto tiles)  
- [tiles_JPG_25p.zip](https://huggingface.co/datasets/abturjo/dengue_test_data/resolve/main/tiles_JPG_25p.zip)  
- [tiles_JPG_50p.zip](https://huggingface.co/datasets/abturjo/dengue_test_data/resolve/main/tiles_JPG_50p.zip)  
- [tiles_JPG_75p.zip](https://huggingface.co/datasets/abturjo/dengue_test_data/resolve/main/tiles_JPG_75p.zip)  

> _"p" indicates percent overlap between adjacent tiles._

### Training Sets  
- YOLO‑formatted OD data:  
  - `train/YOLO/`  
  - `train/YOLO_noAugmentation/`  
- Segmentation data: `train/seggpt/`  
  Access via the HF tree view:  
  - https://huggingface.co/datasets/abturjo/dengue_train_data/tree/main/YOLO  
  - https://huggingface.co/datasets/abturjo/dengue_train_data/tree/main/YOLO_noAugmentation  
  - https://huggingface.co/datasets/abturjo/dengue_train_data/tree/main/seggpt  

### Pre‑trained Weights  
- [best_v8m.pt](https://huggingface.co/abturjo/dengue_trained_weights/resolve/main/best_v8m.pt)  
- [best_v11m.pt](https://huggingface.co/abturjo/dengue_trained_weights/resolve/main/best_v11m.pt)  

### Additional Roboflow Datasets  
- **Object Detection (OD):**  
  [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/thesis19/mv_dataset)  
- **Full orthophoto (289 buildings):**  
  [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/thesis19/buildingsegmentation1-wdugm)  
- **52‑image test set (segmentation eval):**  
  [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/thesis19/kpur-seg-ann-1024-compressed-cv2-jnwtg)  

### Original Raw Data (Google Drive)  
Contains drone imagery, 3D meshes, point clouds, and orthophotos:  
[Google Drive Folder](https://drive.google.com/drive/folders/1-QoHqvPDSKQjCdzJNVmwiqN8vVoiSlrz?usp=drive_link)


## 🛠️ Platforms & Tools Used

- **Notebook Environments:** Google Colab, Kaggle Notebooks  
- **Annotation & Hosting:** Roboflow, Labelbox, Hugging Face Datasets & Model Hub  
- **Storage & Collaboration:** Google Drive  
- **Frameworks & Libraries:** PyTorch, YOLOv8/YOLOv11, segGPT, GDAL, Rasterio, scikit‑learn, geopandas, MATLAB


## ⚙️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/dengue-breeding-site-detection.git
   cd dengue-breeding-site-detection
   ```

2. **Create a Python environment**

   ```bash
   pip install -r requirements.txt
   # or, if using Conda:
   # conda env create -f environment.yml && conda activate dengue-detector
   ```

3. **Download data & weights**

   * Use Notebooks for SegGPT and YOLO (Both detection and segmentation)
   * Availability of test-set can be found in those notebooks


## ▶️ Running the Pipelines

**Using model.ipynb to run the project**
   
This notebook contains the complete workflow including:
   
1. Segmentations to oriented bounding boxes (bbox per slice)
2. Applying sliding windows from the obtained bounding boxes
3. Execution of greedy_nms for comparison with our modified NMM
4. Running the MATLAB script (seg_accuracy.m) for stage-2 distance-based classification
5. Saving classifications using save_classifications.m and storing data in bbox label format (txt files)
6. Generating a final visualization showing risky buildings highlighted in red on the original orthophoto
   

### For a streamlined end-to-end execution of the full dengue breeding site detection pipeline, **find** the [`run.ipynb`](https://regal-shadow-86c.notion.site/Running-Thesis-Pipeline-24c002698fb280c0a5a1dbe70501640a?source=copy_link) **notebook**.

This notebook guides you through loading data, running with model inference data, spatially combining results, and generating final visualizations.


## 📊 Results & Interpretation

* **Balanced Accuracy & Cost Reduction**

  * Two‑stage: **83.6 %** balanced accuracy, **35 %** cost reduction
  * One‑stage: **69.5 %** balanced accuracy

- **Figures**
  * Balanced accuracy (illustrative):  
    <p align="center">
      <img src="Illustrations%20%5Bshowcase%5D/Simple_Balanced_Accuracy.jpg" alt="Balanced Accuracy Illustration" width="500"/>
    </p>
  * Geo‑visualizations:
    <p align="center">
      <img src="Illustrations%20%5Bshowcase%5D/orthophoto_r83_preview.png" alt="Geo Visualization" width="500"/>
    </p>


## 📜 License & Citation

This project is released under the **MIT License**.
If you use this work, please cite:

> Apurbo R. Turjo, Sushmita Paul `Detection of Dengue Breeding Sites in Large-scale Landscapes Plagued by
Unplanned Urban Development from Aerial Imagery and Remote Sensing with
Deep Learning`


## ✉️ Contact & Acknowledgments

**Author:** Apurbo B. Turjo  
**Email:** [turjob44@gmail.com](mailto:turjob44@gmail.com)  
**Advisor:** Prof. Dr. Anindya Iqbal (Bangladesh University of Engineering and Technology)  
**In collaboration with:** Prof. Dr. Manzur Murshed (Deakin University, Australia), Prof. Dr. Sohel Rahman (Bangladesh University of Engineering and Technology), Mashiat Mustaq (Purdue University)


*Thank you for reviewing this work! For any questions, feel free to open an issue or contact me directly.*
