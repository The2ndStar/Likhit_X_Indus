
📥 Dataset Information
This project utilizes the MVTec Anomaly Detection (MVTec AD) dataset, specifically the Transistor category.

Download Link: [MVTec AD Dataset Downloads](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)
Target Category: Transistor (approx. 384 MB)

Data Structure used in this project:
train/good: 231 images used for the Autoencoder training phase.
test/: 69 images across 5 categories (good, bent_lead, cut_lead, damaged_case, misplaced) used for evaluation and PCA/Random Forest classification.


Likhit_X_Indus/
├── data/
│   ├── train/            # 231 'Good' images for training
│   └── test/             # 69 images (good, bent_lead, cut_lead, etc.)
├── CNN_model.py          # The Autoencoder Architecture
├── CNN_train.py          # Training script (MAE Loss + BatchNormalization)
├── CNN_test.py           # Test CNN
├── requirements.txt      # Library dependencies
└── .gitignore            # Prevents uploading large image folders


🛠️ How to Run
1. Setup Environment
    pip install -r requirements.txt

2. Train the CNN
This will train the model to recognize "Good" transistors. It uses MAE Loss and L1 Regularization to ensure high sensitivity to defects.
    python CNN_train.py

📊 Current Results (Benchmark)
Category         :  Avg Reconstruction Error
Good (Baseline)  :          0.0071
Bent Lead        :          0.0090
Cut Lead         :          0.0088
Misplaced        :          0.0211
Damaged Case     :          0.0072
