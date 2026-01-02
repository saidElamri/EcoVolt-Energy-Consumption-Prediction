# EcoVolt Energy Consumption Prediction ⚡

This project implements a **Deep Learning (LSTM)** model to predict future electricity consumption based on historical data. It was developed as part of a technical test to demonstrate proficiency in handling time-series data, normalization, windowing, and neural network modelling with TensorFlow/Keras.

## 📌 Project Overview

**Objective:** Predict the next hour's electrical consumption ($t+1$) based on the previous 24 hours of data.

**Key Technical Steps:**
1.  **Data Preparation:** Temporal sorting and indexing.
2.  **Normalization:** Using `MinMaxScaler` (critical for LSTMs).
3.  **Windowing:** Creating 3D sequences `(Samples, Timesteps, Features)` with a window size of 24h.
4.  **Modeling:** Long Short-Term Memory (LSTM) network architecture.
5.  **Validation:** Chronological split (no shuffling) to prevent data leakage.

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
The easiest way to run this project is using Google Colab, as it handles all dependencies for you.

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Upload the `prediction_model.ipynb` file from this repository.
3.  Run the first cell ("Environment Setup").
4.  When prompted, upload the `dataset.csv` file.
5.  Run all remaining cells to train the model and visualize predictions.

### Option 2: Local Installation
If you prefer to run it locally, ensure you have Python 3.10+ installed.

1.  Clone this repository:
    ```bash
    git clone https://github.com/saidElamri/EcoVolt-Energy-Consumption-Prediction.git
    cd EcoVolt-Energy-Consumption-Prediction
    ```

2.  Install required packages:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```

3.  Place your `dataset.csv` in the root folder or in a `data/` subfolder.

4.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook prediction_model.ipynb
    ```

## 📊 Results
The notebook includes visualizations for:
- **Loss Curves:** Comparing Training vs Validation loss to monitor for overfitting.
- **Prediction:** A final graph overlaying Real Consumption vs Predicted Consumption on the test set.

## 📚 Pedagogical Notes
The notebook contains detailed Markdown cells explaining the "Why" behind key decisions:
- Why temporal order matters.
- Why we normalize for Neural Networks.
- The structure of LSTM input layers.
