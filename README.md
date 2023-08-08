# Cardiovascular Risk Prediction
Cardiovascular diseases (CVDs) are a group of disorders of the heart and blood vessels. 
CVD is a general term for conditions affecting the heart or blood vessels. It's usually associated with a build-up of fatty deposits inside the arteries (atherosclerosis) and an increased risk of blood clots. It can also be associated with damage to arteries in organs such as the brain, heart, kidneys and eyes.


## Table of Content
  * [Problem Statement](#problem-statement)
  * [Objective](#objective)
  * [Dataset](#dataset)
  * [Data Pipeline](#data-pipeline)
  * [Installation](#installation)
  * [Project Structure](#project-structure)
  * [Tools Used](#tools-used)
  * [Performed Model Result](#performed-model-Result)
  * [Project Summary](#project-summary)
  * [Conclusion](#conclusion)


## Problem Statement
* Cardiovascular diseases (CVDs) are the leading cause of death globally.
* An estimated 17.9 million people died from CVDs in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke.
* Over three quarters of CVD deaths take place in low- and middle-income countries.
* Out of the 17 million premature deaths (under the age of 70) due to noncommunicable diseases in 2019, 38% were caused by CVDs.
* Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol.
* **It is important to detect cardiovascular disease as early as possible so that management with counselling and medicines can begin.**


## Objective
The classification goal is to predict whether the patient has a 10-year risk coronary heart disease (CHD) or not.


## Dataset
The dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes. Each attribute is a potential risk factor. These attributes are demographic, behavioral and medical risk factors. 
More details of the dataset can be found in the kaggle website.https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data#CHD_preprocessed.csv


## Data Pipeline
1. EDA: 
    - EDA or Exploratory Data Analysis is the critical process of performing the initial investigation  on the data.  In this initial step we went to look for different features available and tried to uncover their relevance with the target variable, through this we have observed certain trends and dependencies and drawn  certain conclusions from the dataset that will be useful for further processing.
2. Data Processing: 
    - During this stage, we looked for the data types of each feature and  corrected them. After that comes the null value and outlier detection. For the null values imputation we used Mean, Median and Mode technique and for the outlier we used Capping method to handle the outliers without any loss to the data.
3. Feature Engineering: 
    - During this stage, we went on to select the most relevant  features using the chi-square test, information gain, extra trees classifier and next comes the feature scaling in order to bring down all the values in similar range. After that comes the treatment of class imbalance in the target variable that  is done using random oversampling.
4. Model Fitting and Performance Metric: 
    - Since the data is transformed to an appropriate form  therefore, we pass it to different classification models and calculate the metrics based on which we select a final model that could give us better prediction.
    
    
## Installation
This project requires python 3.6 or any other higher versions of python.
This project need software to run this python notebook "Jupyter Notebook" or "Google colab". It is highly recommended that you install the Anaconda distribution of Python or use "Google Colab" https://colab.research.google.com/, which already has most of the above packages and more included.
 

## Project Structure
```
├── README.md
├── Dataset 
│   ├── data_cardiovascular_risk.csv
│
│
├── EDA
│   ├── Numeric & Categoric features
│   ├── Univariate Analysis
│   ├── Bivariate Analysis
│   ├── Multivariate Analysis
│   ├── Data Cleaning
│       ├── Duplicated values
│       ├── NaN/Missing values
│   ├── Treating Skewness
│   ├── Treating Outlier 
│
├── Feature Engineering
│   ├── Encoding
|       ├── Label Encoding
|       ├── One-Hot Encoding
│   ├── Handling Multicollinerity
|       ├── Correlation
|   ├── Feature Selection
|       ├── ExtraTree Classifier
|       ├── Chi-Square Test
|       ├── Iformation Gain
|   ├── Handling Class Imbalance
|       ├── Synthetic Minority Oversampling Technique (SMOTE)
│
├── Model Building
│   ├── Train Test Split
│   ├── Scaling data
│   ├── Model selection
│   ├── Hyperparameter Tunning
│   ├── Model Explainability
|
│   
├── Report
├── Presentation
├── Result
└── Reference
```


## Tools Used
![image](https://user-images.githubusercontent.com/112171582/205482290-ed2f9a20-5bb6-494e-aed4-6a8fb29fdb7e.png)


## Performed Model Result
![image](https://user-images.githubusercontent.com/112171582/208916072-5be82ea2-09b0-47e6-ad1e-587a3f5ebf8d.png)


## Project Summary
Importing necessary libraries and dataset. Then perform EDA to get a clear insight of the each feature, The raw data was cleaned by treating the outliers and null values. Transformation of data was done in order to ensure it fits well into machine learning models. Then finally the cleaned form of data was passed into different models and the metrics were generated to evaluate the model and then we did hyperparameter tuning in order to ensure the correct parameters being passed to the model. Then check all the model_result in order to select final model based on business application.


## Conclusion
In general, it is good practice to track multiple metrics when developing a machine learning model as each highlights different aspects of model performance. However we are dealing with Heathcare data and our data is imbalanced for that perticular reason we are more focusing towards the Recall score and F1 score.

   - We've noticed that XBG Classifier is the stand out performer among all models with an f1-score of 0.908 and recall score of 0.873 on test data. it's safe to say that XGB Classifier provides an good solution to our problem.
   - In case of Logistic regression, We were able to see the maximum f1-score of 0.695.
   - KNN gave us Highest recall score of 0.954.
   - Out of the tree-based algorithms, LGBMClassifier and RandomForestClassifier was alsi providing an optimal solution towards achieving our Objective. We were able to achieve an f1-score of 0.908 and 0.884 respectively.
   - For SVM(Support Vector Machines) Classifier, the f1-score lies around 0.774.

In the Medical domain (**more focus towards the reducing False negative values, as we dont want to mispredict a person safe when he has the risk**) here the recall score is the most importance. KNN, XGB, LGBM Random Forest gave the best recall score 0.954 ,0.873 ,0.866, 0.863 respectively.

Finally, **The models that can be deployed according to our study is KNN Classifier** because it has highest Recall score and It’s okay to classify a healthy person as having 10-year risk of coronary heart disease CHD (false positive) and following up with more medical tests, but it is not definitely okay to miss identifying a dieses patient or classifying a dieses patient as healthy (false negative).
