# Solar Power Forecasting using LSTM

This project implements a deep learning model (LSTM) to forecast solar power generation using inverter-level plant data and weather sensor data.  
It covers data cleaning, preprocessing, feature engineering, sequence creation, and LSTM model training.
Submitted by:

Suryansh Khatri(102215137)

Manik Agarwal (102265006)
---

## Project Structure

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

# Project Workflow (Text-Based Flowcharts)

### **Overall Workflow**

```
Raw Data (Generation + Weather)
        ↓
Data Cleaning
        ↓
Data Merging
        ↓
Feature Engineering
        ↓
Scaling and Sequence Creation (X_sequences, y_targets)
        ↓
LSTM Model Training
        ↓
Model Evaluation (MAE, RMSE)
        ↓
Saved Model (solar_forecasting_lstm_optimized.h5)
```

---

### **LSTM Model Pipeline**

```
Input Sequences (24 timesteps)
            ↓
LSTM Layer 1
            ↓
LSTM Layer 2
            ↓
Dense Layer
            ↓
Predicted Solar Power
```

---

# Steps Performed

### 1. Data Cleaning & Merging
- Removed duplicates and missing rows  
- Cleaned night-time zero-generation entries  
- Standardized timestamps  
- Output: **final_cleaned_dataset.csv**

---

### 2. Feature Engineering
- Added hour, day, month  
- Generated time-based patterns  
- Scaled features  
- Created: **X_sequences.npy**, **y_targets.npy**

---

### 3. LSTM Model Training
- Used stacked LSTM layers  
- Tuned batch size, learning rate, epochs  
- Performance:
  - **MAE ≈ 180**
  - **RMSE ≈ 230**

Model saved as:  
`models/solar_forecasting_lstm_optimized.h5`

---

# How to Run the Project

### Install Dependencies
```bash
pip install numpy pandas scikit-learn tensorflow
```

### Load and Use the Trained Model
```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("models/solar_forecasting_lstm_optimized.h5")

X = np.load("X_sequences.npy")
prediction = model.predict(X[:1])
print(prediction)
```

---

# Results

| Metric | Value |
|--------|--------|
| **MAE** | ~180 kW |
| **RMSE** | ~230 kW |

The model successfully captures daily generation patterns and performs well on real-world plant data.

---

# Key Features

- End-to-end solar forecasting pipeline  
- Cleaned & processed real plant dataset  
- Time-series sequence generation  
- LSTM-based forecasting model  
- Pre-saved training outputs  
- Reproducible Jupyter notebooks  

---

# Contributors

This project was created as part of a Solar Power Forecasting microproject.  
Responsibilities included:
- Data preprocessing  
- Deep learning model training  
- Documentation  

---

# License

This project is available for academic and learning usage.

---

