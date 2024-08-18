# Breast_Cancer_Prediction
This is my Machine Learning Project for the Master Program at University of South Florida

# Objective

This project aims to identify the best machine learning model for predicting breast cancer using the BC Data dataset. The project involves data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation.

## Background

Most types of breast cancer are easy to diagnose by microscopic analysis of a sample - or biopsy - of the
affected area of the breast. The two most commonly used screening methods, physical examination of the
breasts by a healthcare provider and mammography, can offer an approximate likelihood that a lump is cancer,
and may also detect some other lesions, such as a simple cyst. When these examinations are inconclusive, a
healthcare provider can remove a sample of the fluid in the lump for microscopic analysis (a procedure known
as fine needle aspiration, or fine needle aspiration, FNA) to help establish the diagnosis. A needle aspiration
can be performed in a healthcare provider's office or clinic. Together, physical examination of the breasts,
mammography, and FNA can be used to diagnose breast cancer with a good degree of accuracy.
The features for this dataset are computed from a digitized image of a fine needle aspirate (FNA) of a breast
mass. They describe characteristics of the cell nuclei present in the image. I will use this dataset, ML techniques and python to determine
which model has the highest Recall score, which is to say the model that finds the most True Positives. 

## Data Description

The dataset used in this project includes various attributes related to breast cancer diagnosis. The columns in the dataset are as follows:

- **ID**: A unique identifier for each patient.
- **Clump_Thickness**: Describes the thickness of cell clumps.
- **Uniformity_of_Cell_Size**: Describes the uniformity in the size of cells.
- **Uniformity_of_Cell_Shape**: Describes the uniformity in the shape of cells.
- **Marginal_Adhesion**: Measures the adhesion of cells.
- **Single_Epithelial_Cell_Size**: Describes the size of the epithelial cells.
- **Bare_Nuclei**: Contains numeric data (though initially stored as an object, likely requiring conversion to numeric).
- **Bland_Chromatin**: Measures the texture of the chromatin in the cell nuclei.
- **Normal_Nucleoli**: Describes the condition of the nucleoli in cells.
- **Mitoses**: Measures the number of mitotic figures.
- **Class**: The target variable indicating the diagnosis (2 for benign and 4 for malignant).

## Methodology

The methodology for this project involves several key steps:

1. **Data Preprocessing**:
   - **Handling Missing Values**: Missing values in the `Bare_Nuclei` column were filled with the mean of the column.
   - **Feature Scaling**: Applied `RobustScaler` to ensure features are on a similar scale, which is critical for models like logistic regression and K-nearest neighbors.
   - **Correlation Analysis**: Dropped highly correlated features (`Uniformity_of_Cell_Size` and `Mitoses`) to avoid multicollinearity issues.

2. **Model Development**:
   - **Logistic Regression**: Conducted grid search for hyperparameter tuning, evaluated the model using accuracy, cross-validation score, and ROC-AUC score.
   - **RandomForestClassifier**: Implemented hyperparameter tuning using grid search, evaluated the model's performance, and plotted the ROC curve.
   - **KNeighborsClassifier**: Performed grid search for optimal hyperparameters, and evaluated the model using accuracy, cross-validation score, and ROC-AUC score.

3. **Model Evaluation**:
   - Evaluated models using metrics such as accuracy, cross-validation score, ROC-AUC score, and classification reports.
   - Plotted ROC curves for visual comparison of model performance.
   - Generated confusion matrices to understand the model predictions better.

## Key Features and Insights

### **Feature Engineering**:
- **Correlation Analysis**: Dropped highly correlated features to reduce multicollinearity.
- **Scaling**: Applied `RobustScaler` to ensure features are on a similar scale.

### **Model Performance Summary**

- **Logistic Regression**:
  - **Best Parameters**: `solver='liblinear', C=1, penalty='l1'`
  - **Accuracy**: [Provide the accuracy score here]
  - **Cross Validation Score**: [Provide the cross-validation score here]
  - **ROC-AUC Score**: [Provide the ROC-AUC score here]

- **RandomForestClassifier**:
  - **Best Parameters**: `n_estimators=17, max_features='sqrt', max_depth=4, min_samples_split=5, min_samples_leaf=2, bootstrap=False`
  - **Accuracy**: [Provide the accuracy score here]
  - **Cross Validation Score**: [Provide the cross-validation score here]
  - **ROC-AUC Score**: [Provide the ROC-AUC score here]

- **KNeighborsClassifier**:
  - **Best Parameters**: `n_neighbors=100, algorithm='ball_tree', weights='distance'`
  - **Accuracy**: [Provide the accuracy score here]
  - **Cross Validation Score**: [Provide the cross-validation score here]
  - **ROC-AUC Score**: [Provide the ROC-AUC score here]

### **Comparison to Baseline**

After evaluating different models, **Logistic Regression** emerged as the best model for this dataset. The ROC-AUC score was the highest, and the False Positive Rate (FPR) was very low, indicating a strong model for breast cancer prediction.

## Research Questions and Answers

**Q1. What is the best model for predicting breast cancer from the BC Data?**

- **Answer**: The **Logistic Regression** model with `solver='liblinear', C=1, penalty='l1'` was identified as the best model for predicting breast cancer. It outperformed other models in terms of ROC-AUC score and accuracy.

**Q2. How do different machine learning models compare in terms of accuracy and ROC-AUC score?**

- **Answer**: The Logistic Regression model provided the best ROC-AUC score, indicating its superior performance compared to RandomForestClassifier and KNeighborsClassifier. While RandomForestClassifier and KNeighborsClassifier also performed well, they did not match the Logistic Regression model's performance in terms of both accuracy and ROC-AUC.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any enhancements or bug fixes.

## Acknowledgments

- **Akanksha Kushwaha** for project submission.
- **Scikit-learn Documentation** for guidance on model implementation and evaluation.
