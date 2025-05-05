Thought for a second


```markdown
# Detection of Dengue Breeding Sites in Large‑scale Landscapes Plagued by Unplanned Urban Development from Aerial Imagery and Remote Sensing with Deep Learning

Early detection of dengue breeding sites is crucial for controlling outbreaks in dense urban regions where manual inspections aren’t scalable or cost‑effective. This repository implements a novel two‑stage deep learning pipeline on aerial imagery and remote sensing data to (1) segment individual buildings, (2) detect potential dengue breeding objects, and (3) spatially combine the two for geo‑targeted inspection. In orthophotos from two dengue‑prone districts in Dhaka, our two‑stage method achieved an 83.6 % balanced accuracy with a 35 % reduction in manual inspection cost, outperforming a one‑stage baseline (69.5 % balanced accuracy).

---

## 📂 Repository Structure

```

├── data/
│   ├── train/
│   │   ├── YOLO/                  # Training images & labels for object detection
│   │   ├── YOLO\_noAugmentation/   # Non‑augmented OD training set
│   │   └── seggpt/                # Training set for building segmentation
│   └── test/
│       ├── tiles\_JPG\_25p.zip      # 25 % overlap orthophoto tiles
│       ├── tiles\_JPG\_50p.zip      # 50 % overlap orthophoto tiles
│       └── tiles\_JPG\_75p.zip      # 75 % overlap orthophoto tiles
├── notebooks/
│   ├── orthophoto\_tiles\_division\_(kaggle).ipynb   # Slice raw orthophotos into tiles
│   ├── train\_building\_segmentation.ipynb          # Segmentation model training
│   ├── train\_breeding\_site\_detection.ipynb        # OD model training
│   ├── inference\_and\_evaluation.ipynb             # Run inference & compute metrics
│   └── combine\_and\_visualize\_results.ipynb        # Distance‑based classification & viz
├── src/
│   ├── data\_utils.py           # loading, slicing, augmentation
│   ├── models/                 # model definitions
│   ├── train.py                # CLI entrypoint for training
│   └── infer.py                # CLI entrypoint for inference
├── results/
│   ├── figures/                # ROC curves, confusion matrices, maps
│   └── tables/                 # Balanced accuracy and cost reduction metrics
├── models/
│   ├── best\_v8m.pt             # YOLO‑v8‑medium weights (breeding site detector)
│   └── best\_v11m.pt            # YOLO‑v11‑medium weights
├── output/                     # final geolocation targets & visualization overlays
├── Dockerfile                  # (optional) container setup
├── requirements.txt            # Python (≥3.8) dependencies
└── README.md

````

---

## 🔗 Data & Pre-trained Weights

### Test Sets (orthophoto tiles)  
- [tiles_JPG_25p.zip](https://huggingface.co/datasets/abturjo/dengue_test_data/resolve/main/tiles_JPG_25p.zip)  
- [tiles_JPG_50p.zip](https://huggingface.co/datasets/abturjo/dengue_test_data/resolve/main/tiles_JPG_50p.zip)  
- [tiles_JPG_75p.zip](https://huggingface.co/datasets/abturjo/dengue_test_data/resolve/main/tiles_JPG_75p.zip)  

> _“p” indicates percent overlap between adjacent tiles._

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

---

## 🛠️ Platforms & Tools Used

- **Notebook Environments:** Google Colab, Kaggle Notebooks  
- **Annotation & Hosting:** Roboflow, Labelbox, Hugging Face Datasets & Model Hub  
- **Storage & Collaboration:** Google Drive  
- **Frameworks & Libraries:** PyTorch, YOLOv8/YOLOv11, segGPT, GDAL, Rasterio, scikit‑learn, geopandas

---

## ⚙️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/dengue-breeding-site-detection.git
   cd dengue-breeding-site-detection
````

2. **Create a Python environment**

   ```bash
   pip install -r requirements.txt
   # or, if using Conda:
   # conda env create -f environment.yml && conda activate dengue-detector
   ```

3. **(Optional) Build Docker image**

   ```bash
   docker build -t dengue-detector .
   ```

4. **Download data & weights**

   * Unzip test sets into `data/test/`.
   * Download and place training folders under `data/train/`.
   * Place `best_v8m.pt` and `best_v11m.pt` into `models/`.

---

## ▶️ Running the Pipelines

1. **Slice orthophotos into tiles**

   ```bash
   # In Kaggle or Colab, open:
   notebooks/orthophoto_tiles_division_(kaggle).ipynb
   ```

2. **Train models**

   ```bash
   # Segmentation model
   python src/train.py --config configs/segmentation.yaml  
   # Breeding‑site detection (YOLO)
   python src/train.py --config configs/od.yaml  
   ```

3. **Inference & evaluation**

   ```bash
   python src/infer.py --weights models/best_v8m.pt --data data/test/tiles_JPG_50p/ --output results/
   ```

4. **Distance‑based classification & visualization**

   ```bash
   # Run notebook:
   notebooks/combine_and_visualize_results.ipynb
   ```

---

## 📊 Results & Interpretation

* **Balanced Accuracy & Cost Reduction**
  See `results/tables/metrics.csv` for comparative metrics:

  * Two‑stage: **83.6 %** balanced accuracy, **35 %** cost reduction
  * One‑stage: **69.5 %** balanced accuracy

* **Figures**

  * ROC curves: `results/figures/roc_two_stage_vs_one_stage.png`
  * Geo‑visualizations: `output/inspection_targets_map.png`

---

## 📜 License & Citation

This project is released under the **MIT License**.
If you use this work, please cite:

> Abdur R. Turjo, “Detection of Dengue Breeding Sites in Large‑scale Landscapes…,” *\[Journal/Conference]*, 2025. DOI:10.xxxx/xxxxx

---

## ✉️ Contact & Acknowledgments

**Author:** Apurbo B. Turjo
**Email:** [turjob44@gmail.com](mailto:turjob44@gmail.com)
**Advisor:** Prof. Dr. Anindya Iqbal (Bangladesh University of Engineering and Technology)
**In collaboration with:** Prof. Dr. Manzur Murshed (Deakin University, Australia), Prof. Dr. Sohel Rahman (Bangladesh University of Engineering and Technology), Mashiat Mustaq (Purdue University)

---

*Thank you for reviewing this work! For any questions, feel free to open an issue or contact me directly.*

```
```

