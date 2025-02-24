# Suicidal Text Detection using Machine Learning

## 📝 Overview
This project is a **text classification model** that detects suicidal intent in text data. It uses **NLP techniques** for preprocessing and applies multiple **machine learning models** to classify text as either **suicidal** or **non-suicidal**.

## 📂 Project Workflow
1. **Dataset Handling**  
   - Uses a dataset of text samples labeled as **suicidal (1)** or **non-suicidal (0)**.  
   - Loads data from a CSV file.  
   - Handles missing values and reduces dataset size for efficiency.  

2. **Data Preprocessing**  
   - **Cleaning**: Removes special characters, links, and punctuation.  
   - **Text Normalization**: Converts text to lowercase, corrects grammar, and removes stopwords.  
   - **Stemming & Lemmatization**: Converts words to their root forms.  

3. **Feature Engineering**  
   - **TF-IDF & CountVectorizer**: Converts text data into numerical features for machine learning.  

4. **Model Training & Evaluation**  
   - Trains multiple models:  
     - ✅ **Random Forest**  
     - ✅ **K-Nearest Neighbors (KNN)**  
     - ✅ **Decision Tree**  
     - ✅ **Logistic Regression**  
     - ✅ **Naive Bayes (Multinomial NB)**  
   - **Evaluates models using:**  
     - **Accuracy Score**  
     - **Confusion Matrix**  

5. **Results & Comparison**  
   - Compares model performance using a **bar chart**.  
   - Analyzes class imbalance (suicidal vs. non-suicidal texts).  
   - Displays a **correlation matrix** of features.  

## 🔧 Dependencies
To run the project, install the required libraries:  

```sh
pip install pandas numpy nltk textblob seaborn scikit-learn matplotlib
