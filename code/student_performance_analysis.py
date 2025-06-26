#!/usr/bin/env python3
"""
Student Performance Data Mining Analysis
Author: Muhammet Enes Yigit
GitHub: https://github.com/meygt/student_performance

Classification analysis of student academic performance using machine learning.
Dataset: Student Performance & Behavior Dataset (5,000 records) by Mahmoud Elhemaly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceAnalyzer:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        print("=== LOADING AND PREPROCESSING DATA ===")
        
        # Load the main dataset
        self.data = pd.read_csv('../data/Students Performance Dataset.csv')
        print(f"Dataset loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Basic cleaning
        self.data = self.data.dropna()
        print(f"After removing missing values: {self.data.shape[0]} rows")
        
        # Correct Academic Performance Index based on actual dataset weights
        self.data['Academic_Performance_Index'] = (
            self.data['Midterm_Score'] * 0.15 + 
            self.data['Final_Score'] * 0.25 + 
            self.data['Assignments_Avg'] * 0.15 +
            self.data['Quizzes_Avg'] * 0.10 +
            self.data['Participation_Score'] * 0.05 +
            self.data['Projects_Score'] * 0.30
        )
        
        self.data['Workload_Stress_Ratio'] = (
            self.data['Study_Hours_per_Week'] / (self.data['Sleep_Hours_per_Night'] + 1)
        )
        
        self.data['Engagement_Score'] = (
            self.data['Participation_Score'] * 0.4 + 
            (self.data['Attendance (%)'] / 10) * 0.6
        )
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_dept = LabelEncoder()
        le_extracurr = LabelEncoder()
        le_internet = LabelEncoder()
        le_parent_edu = LabelEncoder()
        le_income = LabelEncoder()
        
        self.data['Gender_Encoded'] = le_gender.fit_transform(self.data['Gender'])
        self.data['Department_Encoded'] = le_dept.fit_transform(self.data['Department'])
        self.data['Extracurricular_Encoded'] = le_extracurr.fit_transform(self.data['Extracurricular_Activities'])
        self.data['Internet_Encoded'] = le_internet.fit_transform(self.data['Internet_Access_at_Home'])
        self.data['Parent_Education_Encoded'] = le_parent_edu.fit_transform(self.data['Parent_Education_Level'])
        self.data['Income_Encoded'] = le_income.fit_transform(self.data['Family_Income_Level'])
        
        print("Data preprocessing completed successfully")
        
    def prepare_features_and_target(self):
        print("\n=== PREPARING FEATURES AND TARGET ===")
        
        # Select features for classification
        feature_columns = [
            'Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 
            'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 
            'Projects_Score', 'Study_Hours_per_Week', 'Stress_Level (1-10)',
            'Sleep_Hours_per_Night', 'Gender_Encoded', 'Department_Encoded',
            'Extracurricular_Encoded', 'Internet_Encoded', 'Parent_Education_Encoded',
            'Income_Encoded', 'Academic_Performance_Index', 'Workload_Stress_Ratio',
            'Engagement_Score'
        ]
        
        X = self.data[feature_columns]
        y = self.data['Grade']
        
        print(f"Features selected: {len(feature_columns)}")
        print(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
    def train_classification_models(self):
        print("\n=== TRAINING CLASSIFICATION MODELS ===")
        
        # Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # Support Vector Machine
        svm = SVC(random_state=42, kernel='rbf')
        svm.fit(self.X_train, self.y_train)
        self.models['SVM'] = svm
        
        print("All models trained successfully")
        
    def evaluate_models(self):
        print("\n=== MODEL EVALUATION ===")
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(self.results[name]['classification_report'])
            
    def create_visualizations(self):
        print("\n=== CREATING VISUALIZATIONS ===")
        
        plt.figure(figsize=(15, 10))
        
        # Accuracy comparison
        plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Confusion matrix for best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        cm = self.results[best_model]['confusion_matrix']
        
        plt.subplot(2, 3, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['A', 'B', 'C', 'D', 'F'],
                   yticklabels=['A', 'B', 'C', 'D', 'F'])
        plt.title(f'Confusion Matrix - {best_model}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Grade distribution
        plt.subplot(2, 3, 3)
        self.data['Grade'].value_counts().sort_index().plot(kind='bar', color='lightcoral')
        plt.title('Grade Distribution')
        plt.xlabel('Grade')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Feature importance (Random Forest)
        if 'Random Forest' in self.models:
            feature_names = [
                'Age', 'Attendance', 'Midterm', 'Final', 'Assignments', 'Quizzes', 
                'Participation', 'Projects', 'Study_Hours', 'Stress', 'Sleep', 
                'Gender', 'Department', 'Extracurricular', 'Internet', 'Parent_Edu', 
                'Income', 'Academic_Index', 'Stress_Ratio', 'Engagement'
            ]
            
            plt.subplot(2, 3, 4)
            importances = self.models['Random Forest'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            plt.bar(range(10), importances[indices])
            plt.title('Top 10 Feature Importance')
            plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45)
            plt.ylabel('Importance')
        
        # Performance metrics comparison
        plt.subplot(2, 3, 5)
        grades = ['A', 'B', 'C', 'D', 'F']
        best_pred = self.results[best_model]['predictions']
        
        pred_dist = pd.Series(best_pred).value_counts().sort_index()
        actual_dist = pd.Series(self.y_test).value_counts().sort_index()
        
        x = np.arange(len(grades))
        width = 0.35
        
        actual_values = [int(actual_dist.get(grade, 0)) for grade in grades]
        pred_values = [int(pred_dist.get(grade, 0)) for grade in grades]
        
        plt.bar(x - width/2, actual_values, width, label='Actual', alpha=0.8)
        plt.bar(x + width/2, pred_values, width, label='Predicted', alpha=0.8)
        
        plt.title('Actual vs Predicted Distribution')
        plt.xlabel('Grade')
        plt.ylabel('Count')
        plt.xticks(x, grades)
        plt.legend()
        
        # Academic performance analysis
        plt.subplot(2, 3, 6)
        grade_performance = self.data.groupby('Grade')['Academic_Performance_Index'].mean()
        grade_performance.plot(kind='bar', color='gold')
        plt.title('Average Performance Index by Grade')
        plt.xlabel('Grade')
        plt.ylabel('Performance Index')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('../results/classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to ../results/classification_analysis.png")
        
    def generate_insights(self):
        print("\n=== KEY INSIGHTS ===")
        
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model]['accuracy']
        
        print(f"1. Best performing model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        # Feature importance analysis
        if 'Random Forest' in self.models:
            feature_names = [
                'Age', 'Attendance', 'Midterm', 'Final', 'Assignments', 'Quizzes', 
                'Participation', 'Projects', 'Study_Hours', 'Stress', 'Sleep', 
                'Gender', 'Department', 'Extracurricular', 'Internet', 'Parent_Edu', 
                'Income', 'Academic_Index', 'Stress_Ratio', 'Engagement'
            ]
            
            importances = self.models['Random Forest'].feature_importances_
            top_features = sorted(zip(feature_names, importances), 
                                key=lambda x: x[1], reverse=True)[:5]
            
            print("\n2. Top 5 most important features for prediction:")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature}: {importance:.4f}")
        
        # Grade distribution analysis
        grade_dist = self.data['Grade'].value_counts(normalize=True).sort_index()
        print(f"\n3. Grade distribution:")
        for grade, percentage in grade_dist.items():
            print(f"   Grade {grade}: {percentage:.2%}")
        
        # Performance correlation
        correlation = self.data['Academic_Performance_Index'].corr(
            self.data['Total_Score']
        )
        print(f"\n4. Academic Performance Index correlation with Total Score: {correlation:.4f}")
        
        print("\n5. Model comparison summary:")
        for model, results in self.results.items():
            print(f"   {model}: {results['accuracy']:.4f} accuracy")

def main():
    print("Student Performance Data Mining Analysis")
    print("By: Muhammet Enes Yigit")
    print("=" * 50)
    
    analyzer = StudentPerformanceAnalyzer()
    
    # Run the complete analysis
    analyzer.load_and_preprocess_data()
    analyzer.prepare_features_and_target()
    analyzer.train_classification_models()
    analyzer.evaluate_models()
    analyzer.create_visualizations()
    analyzer.generate_insights()
    
    print("\n" + "=" * 50)
    print("Analysis completed successfully!")
    print("Check ../results/ folder for visualizations")

if __name__ == "__main__":
    main() 