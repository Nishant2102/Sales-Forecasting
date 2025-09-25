# Sales-Forecasting

A deep learning approach to sales forecasting using CNN, RNN (LSTM/GRU), and MLP models, with a web app interface for inference.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features & Highlights](#features--highlights)  
3. [Repository Structure](#repository-structure)  
4. [Installation & Requirements](#installation--requirements)  
5. [Modeling Details](#modeling-details)  
6. [Dataset / Preprocessing](#dataset--preprocessing)  
7. [Evaluation & Results](#evaluation--results)  
8. [Deployment & Web App](#deployment--web-app)  
9. [Future Work / Roadmap](#future-work--roadmap)  

---

## Project Overview

This project aims to build robust models to forecast future sales, leveraging historical sales data along with additional features such as oil prices, product family, and more. The goal is to capture temporal patterns and complex interactions using neural network architectures like CNN, RNN (LSTM / GRU), and MLP.

A web app is also provided to allow users to interactively make predictions using trained models.

---

## Features & Highlights

- Multiple neural network architectures (CNN, RNN, MLP) for comparative analysis  
- Use of external features (e.g. oil price, product family) to improve forecasting accuracy  
- Model persistence (saving / loading trained models)  
- Web interface to input features and get predictions  
- Modular and extensible code structure  

---

## Repository Structure
Sales-Forecasting/
│

├── datasets/                        ← raw / processed data

├── model-building/                  ← saved models, scripts

│   ├── CNN-model-pkl

│   ├── FNN-model-pkl

│   ├── MLP-model-pkl

│   └── …

├── CNN_RNN_LSTM_MLP.ipynb           ← main experiments notebook

├── Final_Data_creation_train.ipynb  ← data preprocessing & training


├── SalesForecasting-app.py          ← web app for inference

├── README.md                        ← project documentation

└── requirements.txt                 ← dependencies


---

## Installation & Requirements
1. Clone the repository:
   ```bash
   git clone [https://github.com/Nishant2102/Sales-Forecasting.git](https://github.com/Nishant2102/Sales-Forecasting.git)
   cd Sales-Forecasting
Install dependencies:
```bah
pip install -r requirements.txt
Place datasets into the datasets/ folder.
```
Usage
Training / Model Experiments (Notebook)
Run the notebook:
```bash
jupyter notebook CNN_RNN_LSTM_MLP.ipynb
```
## Workflow:
-Load & preprocess data
- Train CNN, RNN, MLP models
- Evaluate performance
- Save best model

### Web App (Inference / Prediction)
Start the app:
```bash
python SalesForecasting-app.py
```
Access it in browser and upload your data to get forecasts.

### Modeling Details
- CNN: captures local temporal patterns
- RNN (LSTM/SimpleRNN): models sequential dependencies
- MLP: a robust baseline model
- Features: date, onpromotion, oil prices
- Persistence: models saved/loaded via pickle or native APIs

### Dataset / Preprocessing
The data used is separated into a training set and a testing set.

### Steps:

- Handle missing values
- Feature engineering (time-based features)
- Scaling/normalization using MinMaxScaler
- Time-based train/test split

### Evaluation & Results
Metrics: Mean Squared Error (MSE), which measures the average squared difference between predictions and actual values.

The Simple RNN model demonstrated the best performance with the lowest MSE on the training data.

### Deployment & Web App
The SalesForecasting-app.py serves predictions with Streamlit.

It loads saved model outputs, accepts a CSV input, and displays the forecasts and related visualizations.

## Future Work / Roadmap
- Hyperparameter tuning & ensembles
- Add more external features (promotions, holidays)
- Forecast uncertainty intervals
- Enhanced web UI with dashboards

  

