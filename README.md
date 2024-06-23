#### CRISPRclassify-CNN-Att

##### Project Description

CRISPRclassify-CNN-Att is a deep learning-based method that utilizes Convolutional Neural Networks (CNNs) and self-attention mechanisms to classify CRISPR-Cas systems based on repeat sequences.

##### File Description

- **data/**: Contains data files .
  - `repeats_all.csv`: CSV file containing all repeats and their corresponding subtypes.
- **source/**: Contains all the script files.
  - `CNN_Att.py`: CNN and self-attention mechanism models.
  - `dataselect.py`: selecting datasets.
  - `repeatsEncoder.py`: encoding repeats.
  - `structure_features.py`: calculating additional features (including k-mer frequency, GC content, and sequence length).
  - `transferlearning.py`: fine-tuning for classifying less abundant subtypes .
  - `typeEncoder.py`: encoding subtypes.
  - `stacking.py`: model stacking.
- `README.md`: Project description and instructions.
- `requirements.txt`: List of dependencies for the project.

##### Installation Guide

1. Clone the repository:
    ```bash
    git clone https://github.com/Xingyu-Liao/CRISPRclassify-CNN-Att.git
    cd CRISPRclassify-CNN-Att
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    

##### Usage Instructions

1. Train model-large for the subtype with abundant samples:

   ```python
   python CNN_Att.py
   ```

2. Train model-less for the subtype with fewer samples:

   ```python
   python transferlearning.py
   ```

3. Model stacking:

   ```python
   python stacking.py
   ```

##### Project Structure

CRISPRclassify-CNN-Att/
├── data/
│   ├── repeats_all.csv
├── source/
│   ├── CNN_Att.py
│   ├── dataselect.py
│   ├── repeatsEncoder.py
│   ├── structure_features.py
│   ├── transferlearning.py
│   ├── typeEncoder.py
│   ├── stacking.py
├── README.md
├── requirements.txt