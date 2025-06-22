# 🏡 Airbnb Price Prediction & Classification
A machine learning project to predict Airbnb listing prices and classify listings into pricing tiers (low, medium, high) using structured data and image features.
Airbnb Property Price Prediction Model

# 🔍 Project Goals

Airbnb Listing project, in which you will develop a multitude of classification & regression models and compare their performance across different use cases. The user will load the Airbnb dataset which contains both numerical & categorical data and perform cleaning transformations on this data. You will address the following:

Regression: Predict nightly price of an Airbnb listing.
Classification: Categorize listings by price level (low vs not low, or multi-class).
Feature Enrichment: Use structured data + image embeddings (ResNet50 + PCA).
Imbalance Handling: Apply SMOTE + class weights for fair training.

# 🧠 Models Used

🔢 Price Prediction (Regression)
Linear Regression
Random Forest
Gradient Boosting (Tuned)
XGBoost
PCA-enhanced models (with image features)

# 🏷️ Price Level Classification

Random Forest Classifier
XGBoost / LightGBM
Neural Network (with SMOTE and class weights)

# 📊 Evaluation Metrics

Regression: MAE, R², RMSE
Classification: Accuracy, Precision, Recall, F1, AUC
Confusion Matrices & ROC Curves included

# ✅ Key Results

Task	Best Model	Notes
Price (regression)	Tuned GB + PCA image features	R² = 0.43, MAE ≈ $62
Class (binary)	Neural Net + SMOTE + Class Weights	Balanced recall (41%) and AUC = 0.64

# 📂 Project Structure

/notebooks           ← EDA, model training
/images              ← Airbnb image folders
/data                ← Cleaned listings and features
/models              ← Saved models (.pkl, .h5)
/outputs             ← ROC curves, confusion matrices
README.md

# 🛠️ Tech Stack
Python (Pandas, Scikit-Learn, XGBoost, LightGBM, Keras)
ResNet50 (TensorFlow) for image feature extraction
Matplotlib, Seaborn for visualizations
