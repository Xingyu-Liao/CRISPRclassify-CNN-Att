#### CRISPRclassify-CNN-Att

##### Project Description

CRISPRclassify-CNN-Att is a deep learning-based method that utilizes Convolutional Neural Networks (CNNs) and self-attention mechanisms to classify CRISPR-Cas systems based on repeat sequences.

##### Sample data

| Repeats                              | Subtype |
| :----------------------------------- | :------ |
| GTCGCGCCTTTACGGGCGCGTGGATTGAAAC      | I-C     |
| CGGTTCATCCCCACCTGCGTGGGGTTAAT        | I-E     |
| GATTGAAAGCTATGCGAATTTGCACAGTCTTAAAAC | VI-D    |
| TCCAGCCGCCTTCAGGCGGCTGGTGTGTTGAAAC   | I-C     |
| ATAAGAGAGAATATAACTCCGATAGGAGACGGAAAC | III-A   |
| GTCTGCCCCGCGCATGCGGGGATGAACCC        | I-E     |

##### File Description

- **data/**: Contains data files .
  - `repeats_all.csv`: csv file containing all repeats and their corresponding subtypes.
  
- **source/**: Contains all the script files.
  - `CNN_Att.py`: CNN and self-attention mechanism models.
  - `dataselect.py`: selecting datasets.
  - `repeatsEncoder.py`: encoding repeats.
  - `structure_features.py`: calculating additional features (including k-mer frequency, GC content, and sequence length).
  - `transferlearning.py`: fine-tuning for classifying less abundant subtypes .
  - `typeEncoder.py`: encoding subtypes.
  - `stacking.py`: model stacking.
  
- model/:
  - cnn_att_large.pth: pre-trained model-large
  
  - cnn_att_less.pth：pre-trained model-less
  
  - CRISPRclassify_CNN_Att.pkl  : stacking model
  
    > [!NOTE]
    >
    > Due to file size limitations, we have stored the files `cnn_att_large.pth`,`cnn_att_less.pth`,and`CRISPRclassify_CNN_Att.pkl` at the following link: [Google Drive Folder](https://drive.google.com/drive/folders/1G5v5eQX1lXrIqmJpp34Kwi0fw1SqvtLe?usp=sharing).
  
  - test.py : model testing
  
  - test.xlsx: test data
  
- `README.md`: project description and instructions.

- `requirements.txt`: list of dependencies for the project.

##### Installation Guide

1. Clone the repository:
    ```bash
    git clone https://github.com/Xingyu-Liao/CRISPRclassify-CNN-Att.git
    cd CRISPRclassify-CNN-Att
    ```

2. Create and activate a virtual environment :
    ```bash
    conda create --name crisprclassify-cnn-att
    conda activate crisprclassify-cnn-att
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

##### Usage Instructions

1. Train model-large for the subtype with abundant samples:

   ```bash
   cd source
   python CNN_Att.py
   ```

2. Train model-less for the subtype with fewer samples:

   ```bash
   cd source
   python transferlearning.py
   ```

3. Model stacking:

   ```bash
   cd source
   python stacking.py
   ```

4. Model testing:

   ```bash
   cd model
   python test.py
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
├── model/  
│   ├── cnn_att_large.pth 
│   ├── cnn_att_less.pth
│   ├── CRISPRclassify_CNN_Att.pkl
│   ├── test.py  
│   ├── test.xlsx  
├── README.md  
├── requirements.txt  