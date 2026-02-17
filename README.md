🚖 Taxi Fare Prediction
Decision Tree & PyTorch Neural Network Implementation
📌 Project Overview

This project builds and compares two machine learning models to predict taxi fare amounts using the NYC Yellow Taxi Trip Data (March 2016) dataset:

🌳 Decision Tree Regressor (Scikit-learn)

🧠 Neural Network (PyTorch)

The objective is to predict fare_amount using trip distance, time features, passenger information, and geographic coordinates.

📊 Dataset Information

Dataset: NYC Yellow Taxi Trip Data (March 2016)
Records: ~12 Million Trips

Key Columns Used

trip_distance

passenger_count

RatecodeID

payment_type

pickup_longitude, pickup_latitude

dropoff_longitude, dropoff_latitude

fare_amount

Pickup datetime features

🧹 Data Cleaning & Feature Engineering
✔ Cleaning Steps

Removed duplicate records

Removed invalid values:

fare_amount <= 0 or fare_amount >= 200

trip_distance <= 0 or trip_distance >= 100

⏳ Time-Based Features Extracted

pickup_hour

pickup_day

pickup_month

These features help capture demand patterns and pricing variations.

🌳 Model 1 — Decision Tree Regressor
⚙ Configuration

max_depth = 10

random_state = 42

Train/Test Split: 80% / 20%

📈 Performance
Metric	Value
MAE	$1.25
R² Score	0.9541
🔍 Feature Importance
Feature	Importance
trip_distance	~94.7%
RatecodeID	~3.7%
Others	< 1%

📌 Insight: Taxi fare is heavily distance-driven.

🧠 Model 2 — PyTorch Neural Network
🏗 Architecture
Input (11 features)
      ↓
Linear (64) + ReLU
      ↓
Linear (32) + ReLU
      ↓
Output (1)

⚙ Training Setup

Loss Function: MSELoss

Optimizer: Adam

Learning Rate: 0.001

Epochs: 100

Batch Size: 32

Feature Scaling: StandardScaler

📊 Evaluation Metrics

Mean Absolute Error (MAE)

R² Score

⚡ Real-World Dynamic Pricing

Modern ride-hailing platforms use advanced pricing strategies based on:

⏰ Time of day (rush hour)

🌧 Weather conditions

🎉 Special events

📍 Real-time supply & demand

🚦 Traffic congestion

📈 Surge pricing algorithms

This project predicts base fare using historical structured data, not real-time surge pricing.

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

PyTorch

KaggleHub

🚀 How to Run
1️⃣ Install Dependencies
pip install pandas numpy scikit-learn torch kagglehub

2️⃣ Download Dataset

Download March 2016 taxi data from Kaggle.

3️⃣ Run Notebook

Execute cells in order:

Data Loading

Cleaning

Feature Engineering

Model Training

Evaluation

📌 Key Takeaways

✔ Distance dominates fare prediction
✔ Decision Trees perform extremely well on tabular data
✔ Neural Networks require scaling & tuning
✔ Cleaning outliers significantly improves performance

🔮 Future Improvements

Add weather API data

Include real-time traffic estimates

Use ensemble models (Random Forest / XGBoost)

Hyperparameter tuning

Deploy as a web application
