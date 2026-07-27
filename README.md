# Employee Attrition Prediction using Decision Tree and Random Forest Classification

## Objective

The objective of this project is to develop Decision Tree and Random Forest classification models to predict employee attrition using the IBM HR Analytics dataset. The models are trained and evaluated to compare their performance using standard classification metrics.

## Dataset Link

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Libraries Used

- pandas
- matplotlib
- scikit-learn

## Methodology

1. Loaded the IBM HR Analytics Employee Attrition dataset using Pandas.
2. Displayed the first five records of the dataset.
3. Identified numerical and categorical features along with the target variable.
4. Displayed dataset information and summary statistics.
5. Checked for missing values.
6. Removed unnecessary columns such as EmployeeCount, EmployeeNumber, Over18, and StandardHours.
7. Encoded categorical variables using LabelEncoder.
8. Split the dataset into 80% training and 20% testing sets.
9. Trained a Decision Tree Classifier.
10. Trained a Random Forest Classifier with 100 estimators.
11. Evaluated both models using Accuracy, Precision, Recall, and F1 Score.
12. Displayed the Confusion Matrix for both models.
13. Generated a Feature Importance plot for the Random Forest model.
14. Compared the performance of both models.

## Results

- Successfully trained both Decision Tree and Random Forest classifiers.
- Compared the models using Accuracy, Precision, Recall, and F1 Score.
- Random Forest generally achieved better prediction performance than the Decision Tree model.
- Feature Importance identified the most influential employee attributes affecting attrition.

## Conclusion

This project successfully implemented Decision Tree and Random Forest classification models to predict employee attrition. The dataset was preprocessed by removing unnecessary columns and encoding categorical variables before training the models. Both models were evaluated using Accuracy, Precision, Recall, and F1 Score, along with Confusion Matrices. The Random Forest model generally performed better because it combines multiple decision trees, reducing overfitting and improving prediction accuracy. Feature Importance analysis helped identify the employee characteristics that contributed most to attrition. Overall, Random Forest proved to be a more reliable model for predicting employee attrition than a single Decision Tree.
