# Spatiotemporal Analysis of Urban Expansion in Greater Vienna (2016–2025) Using Machine Learning Model

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Sentinel-2](https://img.shields.io/badge/Data-Sentinel--2-green?logo=satellite&logoColor=white)](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/)

---

## 🗺️ Project Overview
This project performs a multi-temporal Land Cover Change Detection analysis on the Greater Vienna Region (approx. 3,860 km²) to monitor urban expansion over a 9-year period. Using **Sentinel-2** satellite imagery and a **Random Forest Classifier**, the analysis identifies transitions between vegetation, agriculture, and settlement areas.

The workflow implements a full geospatial pipeline in Python, moving from raw `.SAFE` data to a classified Change Map with **>90% overall accuracy**.

## 📊 Key Findings
* **Urban Expansion:** Detected **548.43 km²** of new settlement area between 2016 and 2025.
* **Primary Loss:** Urban growth occurred primarily at the expense of agricultural land (309 km²) and bare terrain (234 km²).
* **Phenological Impact:** A significant "Agriculture to Terrain" transition was observed, attributed to the seasonal difference between the 2016 epoch (September/Leaf-on) and the 2025 epoch (November/Post-harvest).

## 🖼️ Visual Results
![Change Map Results](Change_Detected.jpg)

## 🛠️ Methodology & Tech Stack

**Technologies:** `Python`, `Rasterio`, `GeoPandas`, `Scikit-Learn`, `NumPy`.

The analysis pipeline consists of four distinct stages:

1.  **Preprocessing:**
    * Automatic recursive search for Sentinel-2 bands (10m resolution).
    * Mosaicking multiple granules to cover the ROI.
    * Masking/Cropping to the study area boundary.
    * **Feature Engineering:** Calculation of NDVI (Normalized Difference Vegetation Index) and stacking as a 5th spectral band.

2.  **Training (Supervised Learning):**
    * Creation of training samples in ArcGIS Pro/GDB.
    * Training a **Random Forest Classifier** (100 trees).
    * Extraction of spectral signatures (Blue, Green, Red, NIR, NDVI) from training polygons.

3.  **Classification & Validation:**
    * Independent models trained for 2016, 2020, and 2025 to account for atmospheric/sensor differences.
    * **Accuracy Assessment:** Achieved >90% Overall Accuracy for all epochs (verified via Confusion Matrix).

4.  **Change Detection:**
    * Post-classification comparison (PCC).
    * Calculation of transition matrices to quantify gains/losses in square kilometers.

## 📂 Repository Structure
```text
├── data/                   # (Ignored by Git) Raw Sentinel-2 images & Intermediates
├── scripts/
│   ├── 01_preprocessing.py # Mosaicking, Masking, and NDVI calculation
│   ├── 02_classification.py# RF Training and Accuracy Assessment
│   └── 03_change_detect.py # Change Matrix and Statistics generation
├── output/                 # CSV Statistics and Accuracy Reports
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation