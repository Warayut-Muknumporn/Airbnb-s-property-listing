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

## Milestone: Data Preparation & Feature Engineering
🎯 Objective:
Prepare and clean the Airbnb listings dataset and associated images for machine learning tasks. Ensure all features (structured and unstructured) are usable and informative.
# ✅ Tasks Completed:
Data Cleaning

Removed duplicates and inconsistent records

Handled missing values (numerical → median, categorical → mode)

Standardized column names and formats

Feature Extraction

Converted list-like columns (e.g., Amenities, Description) from strings to Python lists

Engineered amenities_count as a numeric feature

Normalized numerical columns and converted all types appropriately

Image Processing

Extracted deep image features using ResNet50

Aggregated multiple images per listing (average of embeddings)

Reduced 2048-dim image features to 50D using PCA

Target Variable Creation

Created Price_Level labels: low, medium, high

Engineered binary classification target (low vs not low)

Final Dataset

Merged structured features + image features by listing ID

Saved processed dataset (airbnb_with_image_features.csv) for modeling


## 📌 Milestone: Regression Modeling – Predicting Airbnb Prices

# 🎯 Objective:

Develop and evaluate models that predict Airbnb listing price (Price_Night) using structured and image-based features.

✅ Tasks Completed:
Baseline Modeling

Trained and evaluated Linear Regression as a baseline

Assessed performance with MAE and R²

Advanced Models

Trained:

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

Compared results across models

Image Feature Integration

Combined PCA-reduced image embeddings (from ResNet50)

Evaluated models with and without image features

Hyperparameter Tuning

Used GridSearchCV on Gradient Boosting to find optimal n_estimators, max_depth, and learning_rate

Reduced overfitting and improved MAE

| Model                         | MAE (\$) | R² Score | Notes                         |
| ----------------------------- | -------- | -------- | ----------------------------- |
| Linear Regression             | 86.29    | 0.13     | Basic reference model         |
| Random Forest                 | 80.82    | 0.10     | Slight MAE improvement        |
| Gradient Boosting             | 81.36    | 0.10     | Similar to RF                 |
| XGBoost                       | 69.43    | 0.38     | Better generalization         |
| Tuned GB + PCA Image Features | 62.23    | 0.43     | ✅ Best balance of performance |












