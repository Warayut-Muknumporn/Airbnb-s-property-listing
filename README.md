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

Saved processed dataset (airbnb_with_image_features.csv) for modeling

![Top 10 listing categories](https://github.com/user-attachments/assets/f484a4e6-fe17-4700-a88c-5b4f8cc9b369)
![Distribution of Cleanliness](https://github.com/user-attachments/assets/e57d13fb-ec09-4047-a515-f8c27830b661)
![Distribution of check in rating](https://github.com/user-attachments/assets/aa3e05a2-3d4a-403c-ae5c-2bbc707263c7)
![Distribution of beds](https://github.com/user-attachments/assets/b78a564b-b81e-44a9-8ba3-0cc31479df8c)
![Distribution of bedrooms](https://github.com/user-attachments/assets/c4a4e1ca-5490-4b0b-b207-272017df0c08)
![Distribution of bathrooms](https://github.com/user-attachments/assets/bc93820b-894d-4dbc-a69c-5b9c3b34cf18)
![Distribution of amenities](https://github.com/user-attachments/assets/cd61a778-291a-4022-adf8-56e6a1909890)
![Distribution of Accuracy rating](https://github.com/user-attachments/assets/48702480-832b-42b1-9070-041bd2e43d1d)
![Top 10 Locations](https://github.com/user-attachments/assets/86a7ee82-ae0e-44ab-adac-d4f16f4a1993)
![Top 10 listing categories](https://github.com/user-attachments/assets/4e9356de-197d-4093-b35e-3859c225941c)
![Distribution of Value rat](https://github.com/user-attachments/assets/1bf844c9-e59e-4550-b199-4792d8285680)
![Distribution of Price_Night](https://github.com/user-attachments/assets/881e9234-c7a4-4f48-a119-15406ad5f567)
![Distribution of Location rating](https://github.com/user-attachments/assets/bcb7a0c9-e9af-4533-b254-1bd71a8143db)
![Distribution of guests](https://github.com/user-attachments/assets/06c8ca79-a6f0-4b0d-934f-24ca44a2e679)
![Distribution of cummunication_rating](https://github.com/user-attachments/assets/696ffa9f-866f-4f1d-bf4a-b131d34723a6)



## 📌 Milestone: Data Preparation & Feature Engineering
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


## 📌 Milestone: Classification Modeling – Price Level Detection

# 🎯 Objective:

Classify Airbnb listings into price level categories (low, medium, high) and develop a binary classifier to detect low-priced listings.

✅ Tasks Completed:
Target Engineering

Created Price_Level column using binning:

low (< $75)

medium ($75–$150)

high (> $150)

Created binary target: Low_Binary (1 = low, 0 = not low)

Initial Modeling

Trained baseline Random Forest Classifier

Evaluated with accuracy, precision, recall, F1-score, and confusion matrix

Addressing Class Imbalance

Applied class_weight='balanced' to improve low-class recall

Used SMOTE oversampling on training set

Combined SMOTE + Class Weights in a neural network model for better balance

Advanced Models

Trained:

XGBoost Classifier

LightGBM Classifier

Neural Network (Keras)

Evaluated with classification report and ROC AUC

# Binary Classification Results (Low vs Not Low)

| Strategy                | Accuracy | Low Recall | AUC  | Notes                              |
| ----------------------- | -------- | ---------- | ---- | ---------------------------------- |
| SMOTE Only              | 78%      | 19%        | 0.70 | High accuracy, low minority recall |
| Class Weights Only      | 66%      | 69% ✅      | 0.58 | High recall, poor precision        |
| ✅ SMOTE + Class Weights | 71%      | 41%        | 0.64 | Best balance overall               |

Visualizations

ROC Curves and Confusion Matrices generated for all strategies

Tracked impact of tuning on recall and false positives

🧠 Outcome:
Final model: Neural Network with SMOTE + Class Weights

Best balance between detecting low-price listings and avoiding false positives

Performance suitable for internal alerts, listing analysis, or dashboards

## 📌 Milestone: Neural Network Development

# 🎯 Objective:

Explore deep learning models (Keras) to improve classification performance, especially on imbalanced classes like low-price listings.

✅ Tasks Completed:
Architecture Design

Built configurable feedforward neural networks

Used:

Dense layers with ReLU activations

Dropout regularization

Sigmoid (binary) or softmax (multi-class) output layers

Allowed tuning of:

Layer size

Dropout rate

Activation functions

Optimizers

Model Inputs

Used preprocessed structured features (ratings, counts, location)

Optional inclusion of PCA image embeddings

Applied StandardScaler before training

Imbalance Handling Strategies

Trained 3 neural network variants:

A. SMOTE only

B. Class weights only

C. ✅ SMOTE + class weights (combined strategy)

Training Techniques

Used EarlyStopping to prevent overfitting

Evaluated using:

Validation accuracy/loss

Confusion matrix

ROC curve & AUC

Precision, recall, F1-score

# Best Neural Network Result (Strategy C)

| Metric    | Not Low (0) | Low (1)    |
| --------- | ----------- | ---------- |
| Precision | 0.86        | 0.27       |
| Recall    | 0.77        | **0.41** ✅ |
| Accuracy  | 71%         |            |
| AUC       | 0.64        |            |

Visualizations

ROC Curve for neural model with combined strategy

Confusion matrices comparing A, B, and C

# 🧠 Outcome:
Final model: ✅ Neural Network with SMOTE + Class Weights

Achieved solid balance between recall and generalization

Best suited for internal usage (e.g., flagging underpriced listings)

Ready for deployment or integration in dashboards





