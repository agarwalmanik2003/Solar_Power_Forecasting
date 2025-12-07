# 🌞 Solar Power Forecasting using LSTM

This project builds a deep learning–based system for forecasting solar power generation using inverter-level plant data and weather sensor data.  
It covers data cleaning, merging, feature engineering, sequence preparation, LSTM model training, and saving all required outputs.

---

## 📁 Project Structure

```
Solar_power_forecasting/
│
├── X_sequences.npy
├── y_targets.npy
│
├── data/
│   ├── final_cleaned_dataset.csv
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   ├── Plant_2_Weather_Sensor_Data.csv
│
├── models/
│   └── solar_forecasting_lstm_optimized.h5
│
├── notebooks/
│   ├── PHASE01cleaning_AND_merging.ipynb
│   └── PHASE02_EDA_+_FEATURE_ENGINEERING_FOR_LSTM+phase_3_MODEL_TRAINING.ipynb
│
└── README.md
```

---

## 🔧 Workflow Overview

### ✔ Steps Performed
1. **Data Cleaning & Merging**
   - Removed duplicates & missing values  
   - Cleaned night-time zero-generation rows  
   - Unified timestamps  
   - Output: `final_cleaned_dataset.csv`

2. **Feature Engineering**
   - Added hour, day, month  
   - Irradiance & temperature-based patterns  
   - Scaling applied  
   - Generated supervised sequences:  
     - `X_sequences.npy`  
     - `y_targets.npy`

3. **Model Training (LSTM)**
   - Stacked LSTM layers  
   - Hyperparameter tuning  
   - Performance:
     - **MAE ≈ 180**
     - **RMSE ≈ 230**
   - Saved model:  
     `models/solar_forecasting_lstm_optimized.h5`

---

## 📊 Project Flowchart

### **Overall Workflow**
```mermaid
flowchart TD
    A[Raw CSV Data<br>Generation + Weather] --> B[Data Cleaning]
    B --> C[Merging Datasets]
    C --> D[Feature Engineering]
    D --> E[Scaling & Sequence Creation<br>(X_sequences.npy, y_targets.npy)]
    E --> F[Stacked LSTM Training]
    F --> G[Model Evaluation<br>MAE, RMSE]
    G --> H[Save Model<br>solar_forecasting_lstm_optimized.h5]
```

### **LSTM Model Pipeline**
```mermaid
flowchart LR
    A[Input Sequences<br>(24 timesteps)] --> B[LSTM Layer 1]
    B --> C[LSTM Layer 2]
    C --> D[Dense Layer]
    D --> E[Predicted Solar Power]
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install numpy pandas scikit-learn tensorflow
```

### 2️⃣ Run the Jupyter Notebooks
Located in the `notebooks/` folder:

- PHASE01cleaning_AND_merging.ipynb  
- PHASE02_EDA_+_FEATURE_ENGINEERING_FOR_LSTM+phase_3_MODEL_TRAINING.ipynb  

### 3️⃣ Use the Trained Model for Prediction
```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("models/solar_forecasting_lstm_optimized.h5")

X = np.load("X_sequences.npy")
pred = model.predict(X[:1])
print(pred)
```

---

## 📈 Results

| Metric | Value |
|--------|--------|
| **MAE** | ~180 kW |
| **RMSE** | ~230 kW |

The LSTM model successfully captures daily solar generation patterns and performs well on real-world plant data.

---

## 🧠 Key Features
- Real solar plant dataset  
- Clean, engineered, and preprocessed data  
- LSTM-based time-series forecasting  
- Ready-to-use NPZ sequences  
- Fully trained `.h5` model included  
- Modular notebooks for reproducibility  

---

## 🤝 Contributors
This project was created as part of a Solar Power Forecasting microproject.  
Responsibilities included:
- Data engineering  
- Deep learning model development  
- Documentation preparation  

---

## 📜 License
This project may be used for academic and learning purposes.

---

## ⭐ Support  
If you find this project useful, consider giving the GitHub repository a star ⭐!

