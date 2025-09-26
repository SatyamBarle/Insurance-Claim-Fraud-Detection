import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask Application
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# This is the exact list of columns your model expects
MODEL_COLUMNS = ['months_as_customer', 'policy_deductable', 'umbrella_limit', 
                 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 
                 'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 
                 'injury_claim', 'property_claim', 'vehicle_claim', 'policy_csl_min', 
                 'policy_csl_max', 'insured_sex_MALE', 'insured_education_level_College', 
                 'insured_education_level_High School', 'insured_education_level_JD', 
                 'insured_education_level_MD', 'insured_education_level_Masters', 
                 'insured_education_level_PhD', 'insured_occupation_armed-forces', 
                 'insured_occupation_craft-repair', 'insured_occupation_exec-managerial', 
                 'insured_occupation_farming-fishing', 'insured_occupation_handlers-cleaners', 
                 'insured_occupation_machine-op-inspct', 'insured_occupation_other-service', 
                 'insured_occupation_priv-house-serv', 'insured_occupation_prof-specialty', 
                 'insured_occupation_protective-serv', 'insured_occupation_sales', 
                 'insured_occupation_tech-support', 'insured_occupation_transport-moving', 
                 'insured_relationship_not-in-family', 'insured_relationship_other-relative', 
                 'insured_relationship_own-child', 'insured_relationship_unmarried', 
                 'insured_relationship_wife', 'incident_type_Parked Car', 
                 'incident_type_Single Vehicle Collision', 'incident_type_Vehicle Theft', 
                 'collision_type_Rear Collision', 'collision_type_Side Collision', 
                 'incident_severity_Minor Damage', 'incident_severity_Total Loss', 
                 'incident_severity_Trivial Damage', 'authorities_contacted_Fire', 
                 'authorities_contacted_Other', 'authorities_contacted_Police', 
                 'property_damage_YES', 'police_report_available_YES']

@app.route('/predict', methods=['POST'])
def predict():
    """Receives claim data, preprocesses it, and returns a fraud prediction."""
    # Get the JSON data from the request
    data = request.get_json()
    
    # Convert the JSON to a pandas DataFrame
    input_df = pd.DataFrame(data, index=[0])
    
    # One-hot encode the categorical features from the raw input
    # The 'prefix_sep' and 'columns' must match how the original columns were created
    encoded_df = pd.get_dummies(input_df, 
                                prefix_sep='_', 
                                columns=['insured_sex', 'insured_education_level', 
                                         'insured_occupation', 'insured_relationship', 
                                         'incident_type', 'collision_type', 
                                         'incident_severity', 'authorities_contacted',
                                         'property_damage', 'police_report_available'])
    
    # Align the dataframe columns with the model's columns
    # This adds any missing one-hot encoded columns and fills them with 0
    final_df = encoded_df.reindex(columns=MODEL_COLUMNS, fill_value=0)
    
    # Make a prediction
    prediction = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)
    
    # Format the response
    result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
    confidence = float(prediction_proba[0][prediction[0]])

    return jsonify({
        'prediction': result,
        'confidence_score': confidence
    })

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)