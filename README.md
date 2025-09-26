# Insurance-Claim-Fraud-Detection
End-to-end machine learning project for insurance claim fraud detection — includes EDA, multi-model benchmarking (XGBoost, Random Forest, AdaBoost, etc.), evaluation with ROC-AUC/PR-AUC, and a saved deployable pipeline.
# The workflow covers:
Data cleaning & exploratory analysis
Feature engineering & preprocessing pipeline
Model benchmarking across multiple classifiers
Evaluation using ROC-AUC, PR-AUC, Recall, Precision
Serving predictions via a Flask REST API
 # 1.EDA & Data Cleaning
Handled missing values, categorical encoding, and outliers.
Visualized fraud distribution and patterns across features.
<img width="772" height="637" alt="Screenshot 2025-09-27 000759" src="https://github.com/user-attachments/assets/286ce2c8-8108-43e7-923a-b42e823832d2" />
# 2.Feature Engineering
Derived new flags (e.g., high claim amount, mismatch in claim date vs report date).
Encoded categorical features (OneHot, manual mappings).
# 3.Model Benchmarking
Algorithms tested: XGBoost, Random Forest, AdaBoost, Gradient Boosting, Decision Tree, Voting Classifier
Stratified train/test split for fair evaluation.
 # 4.Evaluation Metrics
XGBoost performed best:
ROC-AUC: 0.79
PR-AUC: 0.72
Recall: 72%
Precision: 68%
Confusion matrix analysis shows trade-off between catching frauds vs false positives.

<img width="753" height="499" alt="Screenshot 2025-09-27 000914" src="https://github.com/user-attachments/assets/bfe1caad-a576-4897-b284-e12f9e07bb8e" />

<img width="766" height="349" alt="Screenshot 2025-09-27 000923" src="https://github.com/user-attachments/assets/ec012cec-c208-4c15-bb92-41a009f139ba" />

# 5.Deployment (API)
Trained model exported as model.pkl
Flask REST API (fraud_detection_api/app.py) accepts JSON input and returns prediction + confidence score

**Setup & Run**
1.Clone Repo
git clone https://github.com/<your-username>/insurance-claim-fraud-detection.git
cd insurance-claim-fraud-detection

2.Install Dependencies
python -m venv venv
.\venv\Scripts\activate   # on Windows
pip install -r requirements.txt

3.Run Flask API
cd fraud_detection_api
python app.py

4.Test API
Invoke-WebRequest -Uri http://127.0.0.1:5000/predict -Method POST -ContentType "application/json" -Body '{
    "months_as_customer": 120,
    "policy_deductable": 1000,
    "umbrella_limit": 0,
    "capital-gains": 50000,
    "capital-loss": -10000,
    "incident_hour_of_the_day": 14,
    "number_of_vehicles_involved": 1,
    "bodily_injuries": 2,
    "witnesses": 2,
    "injury_claim": 7500,
    "property_claim": 5000,
    "vehicle_claim": 45000,
    "insured_sex": "MALE",
    "insured_education_level": "JD",
    "insured_occupation": "craft-repair",
    "insured_relationship": "own-child",
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Rear Collision",
    "incident_severity": "Major Damage",
    "authorities_contacted": "Police",
    "property_damage": "YES",
    "police_report_available": "YES"
}'






