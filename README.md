🚖 Taxi Fare Prediction

Decision Tree & PyTorch Neural Network Implementation

📌 Project Overview

This project builds two machine learning models to predict taxi fare amounts using the NYC Yellow Taxi Trip Data (March 2016) dataset:

Decision Tree Regressor (Scikit-learn)

Neural Network (PyTorch)

The goal is to predict fare_amount based on trip distance, passenger details, time features, and pickup/dropoff locations.

Dataset Source:
NYC Yellow Taxi Trip Data (March 2016)

📊 Dataset Description

The dataset contains over 12 million taxi trip records including:

Trip distance

Passenger count

Pickup & dropoff coordinates

Payment type

Rate code

Fare amount

Timestamp information

🧹 Data Cleaning Steps

Removed duplicate rows

Removed invalid records:

fare_amount <= 0 or fare_amount >= 200

trip_distance <= 0 or trip_distance >= 100

Extracted datetime features:

pickup_hour

pickup_day

pickup_month

🧠 Model 1: Decision Tree Regressor
Features Used

trip_distance

passenger_count

RatecodeID

payment_type

pickup_hour

pickup_day

pickup_month

pickup_longitude

pickup_latitude

dropoff_longitude

dropoff_latitude

Model Configuration

DecisionTreeRegressor

max_depth = 10

random_state = 42

80/20 train-test split

📈 Results

MAE: $1.25

R² Score: 0.9541

🔎 Feature Importance

The most important feature was:

trip_distance (~94.7%)

This shows fare is primarily distance-based.

🧠 Model 2: PyTorch Neural Network
Architecture

Input Layer (11 features)
→ Hidden Layer (64 neurons, ReLU)
→ Hidden Layer (32 neurons, ReLU)
→ Output Layer (1 neuron)

Training Configuration

Loss Function: MSELoss

Optimizer: Adam

Learning Rate: 0.001

Epochs: 100

Batch Size: 32

Feature scaling: StandardScaler

Evaluation Metrics

Mean Absolute Error (MAE)

R² Score

⚡ Dynamic Pricing Insight

Real-world ride-hailing platforms like Uber use much more advanced pricing systems. Factors include:

Time of day (rush hour)

Weather conditions

Special events

Real-time supply & demand

Traffic congestion

Surge pricing algorithms

Our model predicts base fare using historical trip data but does not include live demand-based pricing.

🛠 Technologies Used

Python

Pandas

Scikit-learn

PyTorch

NumPy

🚀 How to Run

Install dependencies:

pip install pandas numpy scikit-learn torch kagglehub


Download the dataset from Kaggle.

Run the notebook cells in order:

Data loading

Cleaning

Feature engineering

Model training

Evaluation

📌 Key Learnings

Distance is the strongest predictor of taxi fare.

Tree-based models perform very well on structured tabular data.

Neural networks require scaling and more tuning.

Data cleaning significantly improves model accuracy.

📚 Future Improvements

Add weather data

Include traffic APIs

Use ensemble models (Random Forest, XGBoost)

Hyperparameter tuning

Deploy as a web app
