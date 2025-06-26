# Student Performance Data Mining Analysis

**Author**: Muhammet Enes Yigit  
**GitHub**: [github.com/meygt/student_performance](https://github.com/meygt/student_performance)

## Project Overview

This project analyzes student academic performance using comprehensive data mining techniques including preprocessing, feature engineering, and machine learning classification. The analysis identifies key factors influencing student success and builds predictive models to enable early intervention strategies.

## Dataset Information

**Source**: Student Performance & Behavior Dataset by Mahmoud Elhemaly  
**Size**: 5,000 real records from a private learning provider  
**Features**: 23 comprehensive attributes covering demographics, academic performance, and behavioral factors

### Dataset Attributes

| Column | Description | Type |
|--------|-------------|------|
| Student_ID | Unique identifier for each student | String |
| First_Name, Last_Name | Student names | String |
| Email | Contact email (anonymized) | String |
| Gender | Male, Female, Other | Categorical |
| Age | Student age | Numerical |
| Department | Student department (CS, Engineering, Business, etc.) | Categorical |
| Attendance (%) | Attendance percentage (0-100%) | Numerical |
| Midterm_Score | Midterm exam score (out of 100) | Numerical |
| Final_Score | Final exam score (out of 100) | Numerical |
| Assignments_Avg | Average assignment scores (out of 100) | Numerical |
| Quizzes_Avg | Average quiz scores (out of 100) | Numerical |
| Participation_Score | Class participation score (0-10) | Numerical |
| Projects_Score | Project evaluation score (out of 100) | Numerical |
| Total_Score | Weighted sum of all grades | Numerical |
| Grade | Letter grade (A, B, C, D, F) | Categorical |
| Study_Hours_per_Week | Average weekly study hours | Numerical |
| Extracurricular_Activities | Participation in extracurriculars (Yes/No) | Categorical |
| Internet_Access_at_Home | Home internet access (Yes/No) | Categorical |
| Parent_Education_Level | Highest parental education (None to PhD) | Categorical |
| Family_Income_Level | Low, Medium, High | Categorical |
| Stress_Level (1-10) | Self-reported stress level | Numerical |
| Sleep_Hours_per_Night | Average nightly sleep hours | Numerical |

### Grade Calculation Formula

**Total Score** = 0.15×Midterm + 0.25×Final + 0.15×Assignments + 0.10×Quizzes + 0.05×Participation + 0.30×Projects

### Dataset Characteristics
- **Missing Values**: Some records contain nulls (Attendance, Assignments, Parent Education)
- **Intentional Bias**: Dataset includes realistic bias patterns for analysis challenges
- **Imbalanced Distributions**: Some departments have more students than others
- **Real Data**: Authentic records with some masking for privacy

## Methodology

### Data Preprocessing
- **Missing Value Handling**: Multiple imputation strategies based on data type
- **Feature Engineering**: Created composite metrics (Academic Performance Index, Workload Stress Ratio, Engagement Score)
- **Data Transformation**: Standardization and categorical encoding
- **Quality Assurance**: Comprehensive data validation and cleaning

### Machine Learning Models
- **Random Forest**: Ensemble method for robust predictions and feature importance
- **Logistic Regression**: Linear baseline model for interpretable results
- **Support Vector Machine**: Non-linear classification with RBF kernel

### Feature Engineering
1. **Academic Performance Index**: Weighted combination following actual grading formula
2. **Workload Stress Ratio**: Study hours relative to sleep recovery time
3. **Engagement Score**: Combined measure of participation and attendance

## Results

### Model Performance
| Model | Accuracy | Characteristics |
|-------|----------|----------------|
| **Logistic Regression** | **97.1%** | Best overall performance, highly interpretable |
| SVM | 92.5% | Strong non-linear pattern recognition |
| Random Forest | 88.3% | Excellent feature importance insights |

### Key Findings
- **Projects (30% weight)** are the strongest predictor of academic success
- **Academic Performance Index** effectively captures overall student capability
- **Engagement factors** (attendance, participation) significantly impact outcomes
- **97.1% prediction accuracy** enables reliable early intervention systems

### Feature Importance Ranking
1. Projects Score (23.3%)
2. Academic Performance Index (22.6%)
3. Final Score (10.5%)
4. Midterm Score (5.4%)
5. Quizzes Average (5.4%)

## Project Structure
```
student_performance_project/
├── data/                          # Dataset files
│   ├── Students Performance Dataset.csv
│   └── Students_Grading_Dataset_Biased.csv
├── code/                          # Analysis implementation
│   └── student_performance_analysis.py
├── reports/                       # Documentation and findings
│   ├── Complete_Data_Mining_Report.md
│   ├── Student_Performance_Data_Preprocessing_Report.pdf
│   └── Student_Performance_Data_Preprocessing_Report.docx
├── results/                       # Generated visualizations
│   └── classification_analysis.png
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
cd code
python3 student_performance_analysis.py
```

### Dependencies
- pandas>=1.5.0
- numpy>=1.24.0
- matplotlib>=3.6.0
- seaborn>=0.11.0
- scikit-learn>=1.2.0

## Applications and Impact

### Educational Benefits
- **Early Warning System**: Identify at-risk students before final assessments
- **Resource Optimization**: Focus support where it's most needed
- **Intervention Strategies**: Data-driven approach to student support
- **Performance Prediction**: Reliable grade forecasting for planning

### Technical Contributions
- **Comprehensive Preprocessing Pipeline**: Handles missing values, bias, and imbalanced data
- **Feature Engineering**: Domain-specific composite metrics
- **Model Comparison**: Multiple algorithms with detailed performance analysis
- **Reproducible Results**: Fixed random seeds and documented methodology

## Future Enhancements
- **Longitudinal Analysis**: Multi-semester tracking
- **Deep Learning Models**: Neural networks for complex pattern recognition
- **Causal Analysis**: Understanding why factors influence performance
- **Real-time Implementation**: Integration with learning management systems

## License and Attribution
- **Dataset**: Original work by Mahmoud Elhemaly
- **Analysis**: Muhammet Enes Yigit
- **Purpose**: Educational data mining and predictive analytics research

---

*This project demonstrates comprehensive data mining methodology applied to educational analytics, achieving high predictive accuracy while providing actionable insights for student success optimization.* 