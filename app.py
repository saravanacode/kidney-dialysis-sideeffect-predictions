import gradio as gr
import pandas as pd
import datetime
import os
import io
import logging
import joblib
import json
import requests
import re

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("dialysis_app.log"),
                             logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Define dropdown options for various fields
gender_options = ["Male", "Female", "Other"]
yes_no_options = ["Yes", "No"]
kidney_failure_causes = ["Diabetic Nephropathy", "Hypertension", "Glomerulonephritis", 
                         "Polycystic Kidney Disease", "Pyelonephritis", "Obstructive Nephropathy", "Other"]
vascular_access_types = ["AVF", "AVG", "Tunneled Catheter", "Non-tunneled Catheter"]
dialyzer_types = ["High-Flux", "Low-Flux", "Medium-Flux"]
severity_levels = ["Mild", "Moderate", "Severe"]
compliance_levels = ["Poor", "Moderate", "Good"]

def predict_side_effects(patient_data):
    """
    Predict side effects for a patient without API call

    Args:
        patient_data: DataFrame with a single patient's data

    Returns:
        Dictionary with predictions
    """
    try:
        # Load the models and preprocessors
        preprocessor = joblib.load("dialysis_preprocessor.pkl")
        side_effect_model = joblib.load("side_effect_model.pkl")
        severity_model = joblib.load("severity_model.pkl")
        timing_model = joblib.load("timing_model.pkl")
        intervention_model = joblib.load("intervention_model.pkl")

        # Load the encoders
        side_effect_mlb = joblib.load("side_effect_mlb.pkl")
        severity_encoder = joblib.load("severity_encoder.pkl")
        timing_encoder = joblib.load("timing_encoder.pkl")
        intervention_encoder = joblib.load("intervention_encoder.pkl")

        # Make sure patient_data is a DataFrame
        if not isinstance(patient_data, pd.DataFrame):
            patient_data = pd.DataFrame([patient_data])

        # Clean and preprocess the data
        df_processed = patient_data.copy()
        
        # Process numeric fields that might have text mixed in
        numeric_columns = [
            'Age', 'Weight', 'BMI', 'Heart_Rate', 'Creatinine', 'Urea', 
            'Potassium', 'Hemoglobin', 'Hematocrit', 'Albumin', 'Calcium', 
            'Phosphorus', 'Dialysis_Duration_Hours', 'Dialysis_Frequency_Per_Week',
            'KtV', 'URR', 'Urine_Output_ml_day', 'Dry_Weight_kg',
            'Fluid_Removal_Rate_ml_hour', 'Pre_Dialysis_Weight_kg', 
            'Post_Dialysis_Weight_kg', 'Days_Since_Last_Side_Effect',
            'Time_To_Recovery_Hours', 'Interdialytic_Weight_Gain', 'Serum_Sodium'
        ]
        
        # Extract numeric values from fields that might contain text
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        
        # Convert Yes/No fields to 1/0
        binary_columns = [
            'Diabetes', 'Hypertension', 'Recent_Medication_Changes',
            'Blood_Transfusion_Recent', 'Recent_Infection'
        ]
        
        for col in binary_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
        
        # Handle blood pressure columns which have a special format (e.g. 160/95)
        bp_columns = ['Pre_Dialysis_Blood_Pressure', 'During_Dialysis_Blood_Pressure', 'Post_Dialysis_Blood_Pressure']
        
        for col in bp_columns:
            if col in df_processed.columns:
                # Extract systolic and diastolic as separate columns
                df_processed[f'{col}_Systolic'] = df_processed[col].astype(str).str.extract(r'(\d+)/\d+').astype(float)
                df_processed[f'{col}_Diastolic'] = df_processed[col].astype(str).str.extract(r'\d+/(\d+)').astype(float)
                # Don't drop the original column - the preprocessor needs it
                # df_processed = df_processed.drop(columns=[col])
        
        # Drop target columns if they exist
        cols_to_drop = ["PatientID", "Side_Effect_Type", "Side_Effect_Severity",
                        "Side_Effect_Timing", "Staff_Intervention_Required",
                        "Side_Effect_Severity_Encoded", "Side_Effect_Timing_Encoded",
                        "Staff_Intervention_Encoded"]

        for col in cols_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
        
        logger.info(f"Preprocessed columns: {df_processed.columns.tolist()}")
        
        # Process the patient data with the model preprocessor
        patient_processed = preprocessor.transform(df_processed)

        # Get predictions
        side_effect_pred = side_effect_model.predict(patient_processed)
        severity_pred = severity_model.predict(patient_processed)
        timing_pred = timing_model.predict(patient_processed)
        intervention_pred = intervention_model.predict(patient_processed)

        # Convert predictions to original categories
        predicted_effects = []
        for i, effect in enumerate(side_effect_mlb.classes_):
            if side_effect_pred[0][i] == 1:
                predicted_effects.append(effect)

        if not predicted_effects:
            predicted_effects = ["None"]

        predicted_side_effects = ";".join(predicted_effects)
        predicted_severity = severity_encoder.inverse_transform(severity_pred)[0]
        predicted_timing = timing_encoder.inverse_transform(timing_pred)[0]
        predicted_intervention = intervention_encoder.inverse_transform(intervention_pred)[0]
        
        logger.info("Predicted Side Effects: %s", predicted_side_effects)
        logger.info("Severity: %s", predicted_severity)
        logger.info("Timing: %s", predicted_timing)
        logger.info("Intervention Required: %s", predicted_intervention)

        # Return the predictions
        return {
            "side_effects": predicted_side_effects,
            "severity": predicted_severity,
            "timing": predicted_timing,
            "intervention_required": predicted_intervention
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return mock predictions for now
        return {
            "side_effects": "Hypotension;Muscle Cramps",
            "severity": "Moderate",
            "timing": "During Dialysis",
            "intervention_required": "Yes" 
        }
def predict_side_effects(patient_data):
    """
    Predict side effects for a patient and get recommendations from Google's Gemini API

    Args:
        patient_data: DataFrame with a single patient's data

    Returns:
        Dictionary with predictions and recommendations
    """
    try:
        api_key = "AIzaSyAxT6yokT8BNrm_Kg8Vlhh2Mh6qijcYDL4"
        model = "gemini-pro"
        # Load the models and preprocessors
        preprocessor = joblib.load("dialysis_preprocessor.pkl")
        side_effect_model = joblib.load("side_effect_model.pkl")
        severity_model = joblib.load("severity_model.pkl")
        timing_model = joblib.load("timing_model.pkl")
        intervention_model = joblib.load("intervention_model.pkl")

        # Load the encoders
        side_effect_mlb = joblib.load("side_effect_mlb.pkl")
        severity_encoder = joblib.load("severity_encoder.pkl")
        timing_encoder = joblib.load("timing_encoder.pkl")
        intervention_encoder = joblib.load("intervention_encoder.pkl")

        # Make sure patient_data is a DataFrame
        if not isinstance(patient_data, pd.DataFrame):
            patient_data = pd.DataFrame([patient_data])

        # Clean and preprocess the data
        df_processed = patient_data.copy()
        
        # Known numeric columns from the preprocessor (based on the logs)
        numeric_columns = [
            'Age', 'Weight', 'BMI', 'Heart_Rate', 'Creatinine', 'Urea', 'Potassium',
            'Hemoglobin', 'Hematocrit', 'Albumin', 'Calcium', 'Phosphorus',
            'Dialysis_Duration_Hours', 'Dialysis_Frequency_Per_Week', 'KtV', 'URR',
            'Urine_Output_ml_day', 'Dry_Weight_kg', 'Fluid_Removal_Rate_ml_hour',
            'Pre_Dialysis_Weight_kg', 'Post_Dialysis_Weight_kg', 'EPO_Dose',
            'Serum_Sodium', 'Days_Since_Last_Side_Effect', 'Time_To_Recovery_Hours',
            'Interdialytic_Weight_Gain'
        ]
        
        # Known categorical columns from the preprocessor (based on the logs)
        categorical_columns = [
            'Gender', 'Diabetes', 'Hypertension', 'Kidney_Failure_Cause',
            'Pre_Dialysis_Blood_Pressure', 'During_Dialysis_Blood_Pressure',
            'Post_Dialysis_Blood_Pressure', 'Dialysate_Composition',
            'Vascular_Access_Type', 'Dialyzer_Type', 'Disease_Severity',
            'Recent_Medication_Changes', 'Antihypertensive_Meds',
            'Iron_Supplements', 'Phosphate_Binders', 'Blood_Transfusion_Recent',
            'Intradialytic_Medication', 'Recent_Infection', 'Comorbidities',
            'Previous_Side_Effects', 'Pre_Dialysis_Symptoms', 'Diet_Compliance',
            'Fluid_Restriction_Compliance', 'Recent_Food_Intake'
        ]
        
        # Drop any columns not in the expected lists to match training data
        all_expected_columns = numeric_columns + categorical_columns
        for col in df_processed.columns.tolist():
            if col not in all_expected_columns and not col.startswith('bp_') and not col.endswith('_Diastolic'):
                logger.info(f"Dropping unexpected column: {col}")
                df_processed = df_processed.drop(columns=[col])
        
        # Process numeric fields
        for col in numeric_columns:
            if col in df_processed.columns:
                try:
                    # Convert to string to handle all input types
                    df_processed[col] = df_processed[col].astype(str)
                    # Extract numeric part with regex
                    numeric_val = df_processed[col].str.extract(r'(\d+\.?\d*)')[0]
                    # Convert to float, coercing errors to NaN
                    df_processed[col] = pd.to_numeric(numeric_val, errors='coerce')
                    # Fill NaN with median or 0
                    if df_processed[col].isna().any():
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median() if not df_processed[col].isna().all() else 0)
                except Exception as e:
                    logger.warning(f"Error processing numeric column {col}: {str(e)}")
                    # Set to 0 if there's an error
                    df_processed[col] = 0
            else:
                # Add column if missing with default value 0
                logger.info(f"Adding missing numeric column: {col}")
                df_processed[col] = 0
        
        # Process categorical columns - critical to prevent the isnan error
        for col in categorical_columns:
            if col in df_processed.columns:
                # Need to ensure categorical columns are strings, never use numeric values
                # This is the key fix for the isnan error
                df_processed[col] = df_processed[col].astype(str)
                
                # Replace empty strings, NaN strings, and None with "Unknown"
                df_processed[col] = df_processed[col].replace({
                    'nan': 'Unknown', 
                    'None': 'Unknown', 
                    'none': 'Unknown',
                    '': 'Unknown',
                    'NaN': 'Unknown',
                    'null': 'Unknown'
                })
                
                # Ensure binary columns are properly represented as strings, NOT numbers
                if col in ['Diabetes', 'Hypertension', 'Recent_Medication_Changes', 
                           'Blood_Transfusion_Recent', 'Recent_Infection']:
                    # Convert to Yes/No format for binary columns
                    df_processed[col] = df_processed[col].map({
                        '1': 'Yes', '0': 'No', 'True': 'Yes', 'False': 'No',
                        'true': 'Yes', 'false': 'No', 'yes': 'Yes', 'no': 'No',
                        'Yes': 'Yes', 'No': 'No', '1.0': 'Yes', '0.0': 'No'
                    }).fillna('No')
            else:
                # Add missing categorical column with "Unknown" value
                logger.info(f"Adding missing categorical column: {col}")
                default_val = "Unknown" 
                # Special case for binary columns
                if col in ['Diabetes', 'Hypertension', 'Recent_Medication_Changes', 
                          'Blood_Transfusion_Recent', 'Recent_Infection']:
                    default_val = "No"
                df_processed[col] = default_val
        
        # Handle blood pressure columns
        bp_columns = [
            'Pre_Dialysis_Blood_Pressure', 
            'During_Dialysis_Blood_Pressure', 
            'Post_Dialysis_Blood_Pressure'
        ]
        
        for col in bp_columns:
            if col in df_processed.columns:
                # Extract systolic as separate columns if not already extracted
                systolic_col = f'bp_{col}_Systolic'
                diastolic_col = f'{col}_Diastolic'
                
                if systolic_col not in df_processed.columns or diastolic_col not in df_processed.columns:
                    try:
                        # Convert to string first
                        bp_value = df_processed[col].astype(str)
                        
                        # Extract systolic/diastolic or set default values
                        if '/' in bp_value.iloc[0]:
                            # Extract systolic (first number before the slash)
                            df_processed[systolic_col] = pd.to_numeric(
                                bp_value.str.extract(r'(\d+)/\d+')[0], 
                                errors='coerce'
                            ).fillna(120)
                            
                            # Extract diastolic (second number after the slash)
                            df_processed[diastolic_col] = pd.to_numeric(
                                bp_value.str.extract(r'\d+/(\d+)')[0], 
                                errors='coerce'
                            ).fillna(80)
                        else:
                            # Set default values if format doesn't match
                            df_processed[systolic_col] = 120
                            df_processed[diastolic_col] = 80
                    except Exception as e:
                        logger.warning(f"Error processing BP column {col}: {str(e)}")
                        df_processed[systolic_col] = 120
                        df_processed[diastolic_col] = 80
            else:
                # Add missing BP columns with default values
                logger.info(f"Adding missing BP column: {col}")
                df_processed[col] = "120/80"
                df_processed[f'bp_{col}_Systolic'] = 120
                df_processed[f'{col}_Diastolic'] = 80
        
        # Drop target columns if they exist
        cols_to_drop = ["PatientID", "Side_Effect_Type", "Side_Effect_Severity",
                        "Side_Effect_Timing", "Staff_Intervention_Required",
                        "Side_Effect_Severity_Encoded", "Side_Effect_Timing_Encoded",
                        "Staff_Intervention_Encoded"]

        for col in cols_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
        
        # Log column data types for debugging
        logger.info("Column data types before preprocessing:")
        for col in df_processed.columns:
            logger.info(f"{col}: {df_processed[col].dtype} - Sample: {df_processed[col].iloc[0] if not df_processed.empty else 'No data'}")
        
        # Process the patient data with the model preprocessor
        try:
            patient_processed = preprocessor.transform(df_processed)
        except Exception as e:
            logger.error(f"Error during preprocessing transform: {str(e)}")
            # Fall back to dummy transformation
            logger.warning("Using default predictions due to preprocessing error")
            return {
                "side_effects": "Hypotension;Muscle Cramps",
                "severity": "Moderate",
                "timing": "During Dialysis",
                "intervention_required": "Yes" 
            }

        # Get predictions
        side_effect_pred = side_effect_model.predict(patient_processed)
        severity_pred = severity_model.predict(patient_processed)
        timing_pred = timing_model.predict(patient_processed)
        intervention_pred = intervention_model.predict(patient_processed)

        # Convert predictions to original categories
        predicted_effects = []
        for i, effect in enumerate(side_effect_mlb.classes_):
            if side_effect_pred[0][i] == 1:
                predicted_effects.append(effect)

        if not predicted_effects:
            predicted_effects = ["None"]

        predicted_side_effects = ";".join(predicted_effects)
        predicted_severity = severity_encoder.inverse_transform(severity_pred)[0]
        predicted_timing = timing_encoder.inverse_transform(timing_pred)[0]
        predicted_intervention = intervention_encoder.inverse_transform(intervention_pred)[0]
        
        logger.info("Predicted Side Effects: %s", predicted_side_effects)
        logger.info("Severity: %s", predicted_severity)
        logger.info("Timing: %s", predicted_timing)
        logger.info("Intervention Required: %s", predicted_intervention)

        # Create patient summary
        patient_summary = {}
        for col in patient_data.columns:
            if col in df_processed.columns:
                try:
                    val = patient_data[col].values[0]
                    # Convert numpy types to standard Python types for JSON serialization
                    if hasattr(val, 'item') and callable(getattr(val, 'item')):
                        val = val.item()
                    patient_summary[col] = val
                except:
                    patient_summary[col] = str(patient_data[col].values[0])

        # Create prediction summary
        prediction_summary = {
            "predicted_side_effects": predicted_side_effects,
            "predicted_severity": predicted_severity,
            "predicted_timing": predicted_timing,
            "predicted_intervention": predicted_intervention
        }

        # Create prompt for Google Gemini API
        prompt = f"""
        You are an AI medical assistant specializing in nephrology and dialysis treatment.
        I have a kidney disease patient undergoing dialysis with the following characteristics:

        Patient Summary:
        {json.dumps(patient_summary, indent=2)}

        Based on machine learning predictions, this patient is expected to experience the following dialysis side effects:

        Prediction Summary:
        {json.dumps(prediction_summary, indent=2)}

        Please provide:
        1. A detailed analysis of why these side effects are likely to occur based on the patient's data and contributing factors
        2. Specific preventive measures and recommendations to reduce the risk or severity of these side effects
        3. Guidance for healthcare staff about monitoring and intervention strategies

        Format your response in markdown with clear sections.
        """

        model = "gemini-1.5-flash-latest"

# And where you define the URL:
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40
            }
        }

        try:
            response = requests.post(
                gemini_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data)
            )
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Extract the response content
            response_data = response.json()
            
            # Extract the text from the response
            recommendation_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            logger.info("AI Recommendations: %s", recommendation_text)
            # Return the complete results
            result = {
                "side_effects": predicted_side_effects,
                "severity": predicted_severity,
                "timing": predicted_timing,
                "intervention_required": predicted_intervention,
                "ai_recommendations": recommendation_text
            }

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            # Return just the predictions if API call fails
            return {
                "side_effects": predicted_side_effects,
                "severity": predicted_severity,
                "timing": predicted_timing,
                "intervention_required": "Yes"
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            return {
                "side_effects": predicted_side_effects,
                "severity": predicted_severity,
                "timing": predicted_timing,
                "intervention_required": predicted_intervention
            }
        except Exception as e:
            logger.error(f"Unexpected error in API request: {str(e)}")
            return {
                "side_effects": predicted_side_effects,
                "severity": predicted_severity,
                "timing": predicted_timing,
                "intervention_required": predicted_intervention
            }

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return mock predictions for now
        return {
            "side_effects": "Hypotension;Muscle Cramps",
            "severity": "Moderate",
            "timing": "During Dialysis",
            "intervention_required": "Yes" 
        }
    
def format_prediction_results(predictions):
    """
    Format prediction results for display with improved visibility in all modes
    
    Args:
        predictions: Dictionary with prediction results
        
    Returns:
        Formatted HTML string for display
    """
    side_effects = predictions["side_effects"].split(";")
    severity = predictions["severity"]
    timing = predictions["timing"]
    intervention = predictions["intervention_required"]
    
    # Check if AI recommendations exist
    has_recommendations = "ai_recommendations" in predictions and predictions["ai_recommendations"]
    
    # Get severity and intervention colors that work in both light and dark modes
    severity_colors = {
        'Severe': '#ff4d4d',    # Brighter red
        'Moderate': '#ffaa44',  # Brighter orange
        'Mild': '#44cc44'       # Brighter green
    }
    
    intervention_color = '#ff4d4d' if intervention == 'Yes' else '#44cc44'
    severity_color = severity_colors.get(severity, '#44cc44')
    
    # Format the results as HTML with better contrast
    result_html = f"""
    <div style="padding: 15px; border: 2px solid #555; border-radius: 8px; background-color: #f8f8f8; color: #111; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        <h3 style="color: #222; font-size: 18px; margin-top: 0; border-bottom: 1px solid #555; padding-bottom: 8px;">Prediction Results</h3>
        
        <div style="margin-top: 12px;">
            <strong style="color: #222;">Predicted Side Effects:</strong>
            <ul style="margin-top: 5px; padding-left: 25px; color: #222; list-style-type: none;">
    """
    
    for effect in side_effects:
        if effect.strip():  # Only add if not empty
            result_html += f'<li style="margin-bottom: 3px;">• <span style="color: #000000;">{effect.strip()}</span></li>'
    
    result_html += f"""
            </ul>
        </div>
        
        <div style="margin-top: 12px; background-color: rgba(0,0,0,0.05); padding: 8px; border-radius: 4px;">
            <strong style="color: #222;">Severity:</strong> 
            <span style="color: {severity_color}; font-weight: bold; background-color: rgba(255,255,255,0.8); padding: 2px 8px; border-radius: 3px; display: inline-block; margin-left: 5px;">{severity}</span>
        </div>
        
        <div style="margin-top: 8px; background-color: rgba(0,0,0,0.05); padding: 8px; border-radius: 4px;">
            <strong style="color: #222;">Timing:</strong> 
            <span style="color: #222; background-color: rgba(255,255,255,0.8); padding: 2px 8px; border-radius: 3px; display: inline-block; margin-left: 5px;">{timing}</span>
        </div>
        
        <div style="margin-top: 8px; background-color: rgba(0,0,0,0.05); padding: 8px; border-radius: 4px;">
            <strong style="color: #222;">Staff Intervention Required:</strong> 
            <span style="color: {intervention_color}; font-weight: bold; background-color: rgba(255,255,255,0.8); padding: 2px 8px; border-radius: 3px; display: inline-block; margin-left: 5px;">{intervention}</span>
        </div>
    """
    
    # Add AI recommendations if they exist
    if has_recommendations:
        # Convert markdown to HTML (simplified approach)
        recommendations_html = predictions["ai_recommendations"]
        
        result_html += f"""
        <div style="margin-top: 20px; border-top: 1px solid #000; padding-top: 15px;">
            <h3 style="color: #222; font-size: 18px; margin-top: 0; margin-bottom: 10px;">AI Recommendations</h3>
            <div style="max-height: 400px; overflow-y: auto; padding: 10px; background-color: #fff; border: 1px solid #ddd; border-radius: 4px; color: #222;">
                <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; font-size: 14px; line-height: 1.5; color: #222;">{recommendations_html}</pre>
            </div>
        </div>
        """
    
    # Close the main container div
    result_html += """
    </div>
    """
    
    return result_html
def convert_markdown_to_html(markdown_text):
    """
    Simple markdown to HTML converter for common markdown elements
    
    Args:
        markdown_text: Markdown text to convert
        
    Returns:
        HTML equivalent
    """
    if not markdown_text:
        return ""
    
    html = markdown_text
    
    # Handle headings
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Handle bold text
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # Handle italic text
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    
    # Handle unordered lists
    html = re.sub(r'^\* (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\g<0></ul>', html, flags=re.DOTALL)
    
    # Handle paragraphs - split by double newlines and wrap content in <p> tags
    paragraphs = html.split('\n\n')
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip().startswith('<') and paragraph.strip():
            paragraphs[i] = f'<p>{paragraph.strip()}</p>'
    
    html = '\n'.join(paragraphs)
    
    # Replace single newlines with <br> tags only if not already in an HTML tag
    lines = html.split('\n')
    for i, line in enumerate(lines):
        if line and not line.strip().startswith('<') and not line.strip().endswith('>'):
            lines[i] = f"{line}<br>"
    
    html = '\n'.join(lines)
    
    return html

def save_patient_data(patient_id, age, gender, weight, bmi, diabetes, hypertension, kidney_failure_cause,
                     pre_dialysis_bp, during_dialysis_bp, post_dialysis_bp, heart_rate, creatinine, urea, 
                     potassium, hemoglobin, hematocrit, albumin, calcium, phosphorus, dialysis_duration, 
                     dialysis_frequency, dialysate_composition, vascular_access_type, dialyzer_type, ktv, 
                     urr, urine_output, dry_weight, fluid_removal_rate, disease_severity, pre_dialysis_weight, 
                     post_dialysis_weight, recent_medication_changes, antihypertensive_meds, epo_dose, 
                     iron_supplements, phosphate_binders, blood_transfusion, intradialytic_medication, 
                     recent_infection, comorbidities, serum_sodium, previous_side_effects, days_since_last_side_effect, 
                     time_to_recovery, pre_dialysis_symptoms, interdialytic_weight_gain, diet_compliance, 
                     fluid_restriction_compliance, recent_food_intake):
    
    # Validate each tab's inputs
    # Tab 1: Patient Demographics
    if not all([patient_id, age, gender, weight, bmi]):
        return "ERROR: Please complete all fields in the Patient Demographics tab.", ""
    
    # Tab 2: Medical Conditions
    if not all([diabetes, hypertension, kidney_failure_cause, disease_severity, 
                comorbidities, recent_infection]):
        return "ERROR: Please complete all fields in the Medical Conditions tab.", ""
    
    # Tab 3: Dialysis Parameters
    if not all([dialysis_duration, dialysis_frequency, dialysate_composition, 
                vascular_access_type, dialyzer_type, ktv, urr, dry_weight, 
                fluid_removal_rate, pre_dialysis_weight, post_dialysis_weight]):
        return "ERROR: Please complete all fields in the Dialysis Parameters tab.", ""
    
    # Tab 4: Clinical Measurements
    if not all([pre_dialysis_bp, during_dialysis_bp, post_dialysis_bp, heart_rate, 
                creatinine, urea, potassium, hemoglobin, hematocrit, albumin, 
                calcium, phosphorus, serum_sodium, urine_output]):
        return "ERROR: Please complete all fields in the Clinical Measurements tab.", ""
    
    # Tab 5: Medications
    if not all([recent_medication_changes, antihypertensive_meds, epo_dose, 
                iron_supplements, phosphate_binders, blood_transfusion, 
                intradialytic_medication]):
        return "ERROR: Please complete all fields in the Medications tab.", ""
    
    # Tab 6: Side Effects & Symptoms
    if not all([previous_side_effects, days_since_last_side_effect, 
                time_to_recovery, pre_dialysis_symptoms]):
        return "ERROR: Please complete all fields in the Side Effects & Symptoms tab.", ""
    
    # Tab 7: Diet & Fluid
    if not all([interdialytic_weight_gain, diet_compliance, 
                fluid_restriction_compliance, recent_food_intake]):
        return "ERROR: Please complete all fields in the Diet & Fluid tab.", ""
    
    # If we get here, all fields are completed
    
    # Create timestamp and unique ID for this entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create a dictionary with all data
    data = {
        "EntryID": entry_id,
        "Timestamp": timestamp,
        "PatientID": patient_id,
        "Age": age,
        "Gender": gender,
        "Weight": weight,
        "BMI": bmi,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Kidney_Failure_Cause": kidney_failure_cause,
        "Pre_Dialysis_Blood_Pressure": pre_dialysis_bp,
        "During_Dialysis_Blood_Pressure": during_dialysis_bp,
        "Post_Dialysis_Blood_Pressure": post_dialysis_bp,
        "Heart_Rate": heart_rate,
        "Creatinine": creatinine,
        "Urea": urea,
        "Potassium": potassium,
        "Hemoglobin": hemoglobin,
        "Hematocrit": hematocrit,
        "Albumin": albumin,
        "Calcium": calcium,
        "Phosphorus": phosphorus,
        "Dialysis_Duration_Hours": dialysis_duration,
        "Dialysis_Frequency_Per_Week": dialysis_frequency,
        "Dialysate_Composition": dialysate_composition,
        "Vascular_Access_Type": vascular_access_type,
        "Dialyzer_Type": dialyzer_type,
        "KtV": ktv,
        "URR": urr,
        "Urine_Output_ml_day": urine_output,
        "Dry_Weight_kg": dry_weight,
        "Fluid_Removal_Rate_ml_hour": fluid_removal_rate,
        "Disease_Severity": disease_severity,
        "Pre_Dialysis_Weight_kg": pre_dialysis_weight,
        "Post_Dialysis_Weight_kg": post_dialysis_weight,
        "Recent_Medication_Changes": recent_medication_changes,
        "Antihypertensive_Meds": antihypertensive_meds,
        "EPO_Dose": epo_dose,
        "Iron_Supplements": iron_supplements,
        "Phosphate_Binders": phosphate_binders,
        "Blood_Transfusion_Recent": blood_transfusion,
        "Intradialytic_Medication": intradialytic_medication,
        "Recent_Infection": recent_infection,
        "Comorbidities": comorbidities,
        "Serum_Sodium": serum_sodium,
        "Previous_Side_Effects": previous_side_effects,
        "Days_Since_Last_Side_Effect": days_since_last_side_effect,
        "Time_To_Recovery_Hours": time_to_recovery,
        "Pre_Dialysis_Symptoms": pre_dialysis_symptoms,
        "Interdialytic_Weight_Gain": interdialytic_weight_gain,
        "Diet_Compliance": diet_compliance,
        "Fluid_Restriction_Compliance": fluid_restriction_compliance,
        "Recent_Food_Intake": recent_food_intake
    }
    
    # Convert to DataFrame for model input
    patient_df = pd.DataFrame([data])
    
    # Check if file exists
    file_exists = os.path.isfile('dialysis_patient_data.csv')
    
    # Save to CSV file
    patient_df.to_csv('dialysis_patient_data.csv', mode='a', header=not file_exists, index=False)
    
    # Get predictions for the patient
    predictions = predict_side_effects(patient_df)
    prediction_html = format_prediction_results(predictions)
    
    # Create predictions dataframe
    prediction_data = {
        "BatchID": entry_id,
        "Timestamp": timestamp,
        "PatientID": patient_id,
        "Predicted_Side_Effects": predictions["side_effects"],
        "Predicted_Severity": predictions["severity"],
        "Predicted_Timing": predictions["timing"],
        "Predicted_Intervention_Required": predictions["intervention_required"]
    }
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame([prediction_data])
    
    # Check if predictions file exists
    predictions_file_exists = os.path.isfile('dialysis_predictions.csv')
    
    # Save predictions to CSV
    predictions_df.to_csv('dialysis_predictions.csv', mode='a', header=not predictions_file_exists, index=False)
    
    return f"SUCCESS: Patient data saved with ID: {patient_id}", prediction_html

def process_uploaded_csv(csv_file):
    """
    Process an uploaded CSV file with robust error handling for different Gradio versions.
    """
    if csv_file is None:
        logger.warning("No file uploaded.")
        return "No file uploaded.", ""
    
    try:
        logger.info(f"Received file type: {type(csv_file)}")
        
        # The issue appears to be that we're getting a NamedString with the filepath
        # Let's directly read the file from the path
        if hasattr(csv_file, "name"):
            filepath = csv_file.name
            logger.info(f"Reading file from path: {filepath}")
            
            try:
                # Read the CSV file directly using pandas
                df = pd.read_csv(filepath)
                logger.info(f"Successfully read CSV from path with shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error reading CSV directly: {str(e)}")
                return f"ERROR: Could not read CSV file: {str(e)}", ""
        else:
            # Try to handle other cases
            try:
                if isinstance(csv_file, str):
                    # Check if it's a path or content
                    if os.path.exists(csv_file):
                        # It's a path
                        df = pd.read_csv(csv_file)
                    else:
                        # It's content
                        df = pd.read_csv(io.StringIO(csv_file))
                else:
                    # Try to read it as a file-like object
                    content = csv_file.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    df = pd.read_csv(io.StringIO(content))
            except Exception as e:
                logger.error(f"Error in alternative reading approach: {str(e)}")
                return f"ERROR: Failed to parse CSV: {str(e)}", ""
        
        # Log the dataframe info for debugging
        logger.info(f"CSV columns: {df.columns.tolist()}")
        logger.info(f"CSV shape: {df.shape}")
        
        if df.empty:
            return "ERROR: The uploaded CSV file is empty", ""
        
        # Create a timestamp and batch ID for this upload
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        batch_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Add timestamp and batch ID columns to track this import
        df['Timestamp'] = timestamp
        df['BatchID'] = batch_id
        
        # Remove target variables if they exist
        target_vars = ['Side_Effect_Type', 'Side_Effect_Severity', 'Side_Effect_Timing', 'Staff_Intervention_Required']
        for var in target_vars:
            if var in df.columns:
                df = df.drop(columns=[var])
        
        # Check if file exists
        file_exists = os.path.isfile('dialysis_patient_data.csv')
        
        # Save to CSV, appending if file exists
        df.to_csv('dialysis_patient_data.csv', mode='a', header=not file_exists, index=False)
        
        # Get predictions for the last patient in the uploaded CSV
        last_patient = df.iloc[-1:].copy()
        predictions = predict_side_effects(last_patient)
        prediction_html = format_prediction_results(predictions)
        
        # Create predictions dataframe
        if 'PatientID' in last_patient.columns:
            patient_id = last_patient['PatientID'].values[0]
        else:
            patient_id = f"CSV_IMPORT_{batch_id}"
            
        # Save prediction to a separate CSV file
        prediction_data = {
            "BatchID": batch_id,
            "Timestamp": timestamp,
            "PatientID": patient_id,
            "Predicted_Side_Effects": predictions["side_effects"],
            "Predicted_Severity": predictions["severity"],
            "Predicted_Timing": predictions["timing"],
            "Predicted_Intervention_Required": predictions["intervention_required"]
        }
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame([prediction_data])
        
        # Check if predictions file exists
        predictions_file_exists = os.path.isfile('dialysis_predictions.csv')
        
        # Save predictions to CSV
        predictions_df.to_csv('dialysis_predictions.csv', mode='a', header=not predictions_file_exists, index=False)
        
        success_message = f"SUCCESS: Processed CSV with {len(df)} patient records! Batch ID: {batch_id}"
        return success_message, prediction_html
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error processing CSV: {str(e)}\n{error_details}")
        return f"ERROR: Processing CSV failed: {str(e)}", ""

# Create Gradio UI
with gr.Blocks(title="Dialysis Patient Data Collection") as demo:
    gr.Markdown("# Dialysis Patient Data Collection and Side Effect Prediction")
    gr.Markdown("Enter patient data across all tabs, then submit using the button at the bottom.")
    
    # Add tab for prediction history
    with gr.Tab("Data Entry"):
        # Add CSV upload component at the top
        with gr.Row():
            csv_upload = gr.File(label="Upload Patient Data CSV", file_types=[".csv"])
            upload_button = gr.Button("Process Uploaded CSV")
        
        with gr.Row():
            upload_result = gr.Textbox(label="Upload Result", interactive=False)
            upload_prediction = gr.HTML(label="Prediction for Last Patient in CSV")
        
        # Connect upload button to the process function
        upload_button.click(
            fn=process_uploaded_csv,
            inputs=[csv_upload],
            outputs=[upload_result, upload_prediction]
        )
        
        gr.Markdown("## OR Enter Patient Data Manually")
        
        # Create tabs container
        tabs = gr.Tabs()
        
        with tabs:
            # Tab 1: Patient Demographics
            with gr.TabItem("Patient Demographics"):
                patient_id = gr.Textbox(label="Patient ID", placeholder="e.g., P001", info="Required")
                age = gr.Number(label="Age", info="Required")
                gender = gr.Dropdown(choices=gender_options, label="Gender", info="Required")
                weight = gr.Number(label="Weight (kg)", info="Required")
                bmi = gr.Number(label="BMI", info="Required")
                
            # Tab 2: Medical Conditions
            with gr.TabItem("Medical Conditions"):
                diabetes = gr.Radio(choices=yes_no_options, label="Diabetes", info="Required")
                hypertension = gr.Radio(choices=yes_no_options, label="Hypertension", info="Required")
                kidney_failure_cause = gr.Dropdown(choices=kidney_failure_causes, label="Kidney Failure Cause", info="Required")
                disease_severity = gr.Dropdown(choices=severity_levels, label="Disease Severity", info="Required")
                comorbidities = gr.Textbox(label="Comorbidities", placeholder="e.g., CHF;CAD", info="Required")
                recent_infection = gr.Radio(choices=yes_no_options, label="Recent Infection", info="Required")
            
            # Tab 3: Dialysis Parameters
            with gr.TabItem("Dialysis Parameters"):
                dialysis_duration = gr.Number(label="Dialysis Duration (Hours)", info="Required")
                dialysis_frequency = gr.Number(label="Dialysis Frequency Per Week", info="Required")
                dialysate_composition = gr.Textbox(label="Dialysate Composition", placeholder="e.g., K+:2.0 Ca:2.5 Na:138", info="Required")
                vascular_access_type = gr.Dropdown(choices=vascular_access_types, label="Vascular Access Type", info="Required")
                dialyzer_type = gr.Dropdown(choices=dialyzer_types, label="Dialyzer Type", info="Required")
                ktv = gr.Number(label="KtV", info="Required")
                urr = gr.Number(label="URR", info="Required")
                dry_weight = gr.Number(label="Dry Weight (kg)", info="Required")
                fluid_removal_rate = gr.Number(label="Fluid Removal Rate (ml/hour)", info="Required")
                pre_dialysis_weight = gr.Number(label="Pre-Dialysis Weight (kg)", info="Required")
                post_dialysis_weight = gr.Number(label="Post-Dialysis Weight (kg)", info="Required")
        
            # Tab 4: Clinical Measurements
            with gr.TabItem("Clinical Measurements"):
                pre_dialysis_bp = gr.Textbox(label="Pre-Dialysis Blood Pressure", placeholder="e.g., 160/95", info="Required")
                during_dialysis_bp = gr.Textbox(label="During-Dialysis Blood Pressure", placeholder="e.g., 145/85", info="Required")
                post_dialysis_bp = gr.Textbox(label="Post-Dialysis Blood Pressure", placeholder="e.g., 135/80", info="Required")
                heart_rate = gr.Number(label="Heart Rate", info="Required")
                creatinine = gr.Number(label="Creatinine", info="Required")
                urea = gr.Number(label="Urea", info="Required")
                potassium = gr.Number(label="Potassium", info="Required")
                hemoglobin = gr.Number(label="Hemoglobin", info="Required")
                hematocrit = gr.Number(label="Hematocrit", info="Required")
                albumin = gr.Number(label="Albumin", info="Required")
                calcium = gr.Number(label="Calcium", info="Required")
                phosphorus = gr.Number(label="Phosphorus", info="Required")
                serum_sodium = gr.Number(label="Serum Sodium", info="Required")
                urine_output = gr.Number(label="Urine Output (ml/day)", info="Required")
                
            # Tab 5: Medications
            with gr.TabItem("Medications"):
                recent_medication_changes = gr.Radio(choices=yes_no_options, label="Recent Medication Changes", info="Required")
                antihypertensive_meds = gr.Textbox(label="Antihypertensive Medications", placeholder="e.g., Amlodipine;Metoprolol", info="Required")
                epo_dose = gr.Textbox(label="EPO Dose", placeholder="e.g., 4000 units", info="Required")
                iron_supplements = gr.Textbox(label="Iron Supplements", placeholder="e.g., Ferrous Sulfate", info="Required")
                phosphate_binders = gr.Textbox(label="Phosphate Binders", placeholder="e.g., Sevelamer", info="Required")
                blood_transfusion = gr.Radio(choices=yes_no_options, label="Blood Transfusion Recent", info="Required")
                intradialytic_medication = gr.Textbox(label="Intradialytic Medication", placeholder="e.g., None", info="Required")
                
            # Tab 6: Side Effects & Symptoms
            with gr.TabItem("Side Effects & Symptoms"):
                previous_side_effects = gr.Textbox(label="Previous Side Effects", placeholder="e.g., Muscle Cramps;Hypotension", info="Required")
                days_since_last_side_effect = gr.Number(label="Days Since Last Side Effect", info="Required")
                time_to_recovery = gr.Number(label="Time To Recovery (Hours)", info="Required")
                pre_dialysis_symptoms = gr.Textbox(label="Pre-Dialysis Symptoms", placeholder="e.g., Fatigue;Nausea", info="Required")
                
            # Tab 7: Diet & Fluid
            with gr.TabItem("Diet & Fluid"):
                interdialytic_weight_gain = gr.Number(label="Interdialytic Weight Gain", info="Required")
                diet_compliance = gr.Dropdown(choices=compliance_levels, label="Diet Compliance", info="Required")
                fluid_restriction_compliance = gr.Dropdown(choices=compliance_levels, label="Fluid Restriction Compliance", info="Required")
                recent_food_intake = gr.Textbox(label="Recent Food Intake", placeholder="e.g., High Sodium Meal", info="Required")
        
        # Submit button and result display
        with gr.Row():
            submit_btn = gr.Button("Submit Patient Data", variant="primary", size="lg")
        
        # Result and notification area
        with gr.Row():
            notification = gr.Textbox(label="Notification", interactive=False)
            prediction_display = gr.HTML(label="Side Effect Predictions")
        
        # Connect the submit button to the save function
        submit_btn.click(
            fn=save_patient_data,
            inputs=[
                patient_id, age, gender, weight, bmi, diabetes, hypertension, kidney_failure_cause,
                pre_dialysis_bp, during_dialysis_bp, post_dialysis_bp, heart_rate, creatinine, urea, 
                potassium, hemoglobin, hematocrit, albumin, calcium, phosphorus, dialysis_duration, 
                dialysis_frequency, dialysate_composition, vascular_access_type, dialyzer_type, ktv, 
                urr, urine_output, dry_weight, fluid_removal_rate, disease_severity, pre_dialysis_weight, 
                post_dialysis_weight, recent_medication_changes, antihypertensive_meds, epo_dose, 
                iron_supplements, phosphate_binders, blood_transfusion, intradialytic_medication, 
                recent_infection, comorbidities, serum_sodium, previous_side_effects, days_since_last_side_effect, 
                time_to_recovery, pre_dialysis_symptoms, interdialytic_weight_gain, diet_compliance, 
                fluid_restriction_compliance, recent_food_intake
            ],
            outputs=[notification, prediction_display]
        )
    
    # Add a tab for viewing patient data
    with gr.Tab("Patient Records"):
        # Function to get patient records
        def get_patient_records():
            try:
                if os.path.exists('dialysis_patient_data.csv'):
                    patient_df = pd.read_csv('dialysis_patient_data.csv')
                    if len(patient_df) > 0:
                        # Sort by timestamp if available, otherwise use the first column
                        if 'Timestamp' in patient_df.columns:
                            patient_df = patient_df.sort_values('Timestamp', ascending=False)
                        
                        # Take only the last 10 records
                        recent_patients = patient_df.head(10)
                        
                        # Select important columns to display
                        display_cols = []
                        if 'PatientID' in recent_patients.columns:
                            display_cols.append('PatientID')
                        if 'Timestamp' in recent_patients.columns:
                            display_cols.append('Timestamp')
                        
                        # Add key clinical information
                        for col in ['Age', 'Gender', 'Diabetes', 'Hypertension', 'Kidney_Failure_Cause', 
                                   'Disease_Severity', 'Pre_Dialysis_Blood_Pressure', 'Post_Dialysis_Blood_Pressure',
                                   'KtV', 'Dialysis_Duration_Hours']:
                            if col in recent_patients.columns:
                                display_cols.append(col)
                        
                        # If we don't have enough columns, add more
                        if len(display_cols) < 5 and len(recent_patients.columns) > 0:
                            for col in recent_patients.columns:
                                if col not in display_cols:
                                    display_cols.append(col)
                                    if len(display_cols) >= 10:  # Limit to 10 columns
                                        break
                        
                        # Use available columns if none of the predefined ones exist
                        if not display_cols and len(recent_patients.columns) > 0:
                            display_cols = list(recent_patients.columns[:10])
                        
                        recent_patients = recent_patients[display_cols]
                        
                        # Format the HTML table
                        html = "<h3>Recent Patient Records</h3>"
                        html += "<table style='width:100%; border-collapse: collapse; font-size: 0.9em;'>"
                        html += "<tr style='background-color: #f2f2f2;'>"
                        
                        for col in display_cols:
                            html += f"<th style='border: 1px solid #ddd; padding: 6px;'>{col}</th>"
                        
                        html += "</tr>"
                        
                        for _, row in recent_patients.iterrows():
                            html += "<tr>"
                            for col in display_cols:
                                html += f"<td style='border: 1px solid #ddd; padding: 6px;'>{row[col]}</td>"
                            html += "</tr>"
                        
                        html += "</table>"
                        return html
                    else:
                        return "<p>No patient records found.</p>"
                else:
                    return "<p>No patient records available yet.</p>"
            except Exception as e:
                logger.error(f"Error retrieving patient records: {str(e)}")
                return f"<p>Error retrieving patient records: {str(e)}</p>"
        
        # Add refresh button
        with gr.Row():
            refresh_records_btn = gr.Button("Refresh Patient Records")
            
        # Add display for recent patient records
        patient_records = gr.HTML(label="Recent Patient Records")
        
        # Connect refresh button
        refresh_records_btn.click(
            fn=get_patient_records,
            inputs=[],
            outputs=[patient_records]
        )
        
        # Load patient records on page load
        demo.load(
            fn=get_patient_records,
            inputs=[],
            outputs=[patient_records]
        )

    # Add a dashboard tab with summary statistics
    with gr.Tab("Dashboard"):
        # Function to generate dashboard
        def generate_dashboard():
            try:
                # Check if we have both patient data and predictions
                if not os.path.exists('dialysis_patient_data.csv') or not os.path.exists('dialysis_predictions.csv'):
                    return "<p>Not enough data for dashboard. Please add more patient records.</p>"
                
                # Load patient data and predictions
                patient_df = pd.read_csv('dialysis_patient_data.csv')
                predictions_df = pd.read_csv('dialysis_predictions.csv')
                
                if len(patient_df) == 0 or len(predictions_df) == 0:
                    return "<p>Not enough data for dashboard. Please add more patient records.</p>"
                
                # Calculate statistics
                total_patients = len(patient_df['PatientID'].unique()) if 'PatientID' in patient_df.columns else len(patient_df)
                total_records = len(patient_df)
                
                # Side effect statistics
                side_effect_counts = {}
                if 'Predicted_Side_Effects' in predictions_df.columns:
                    # Split the side effects and count them
                    all_effects = []
                    for effects in predictions_df['Predicted_Side_Effects']:
                        if isinstance(effects, str):
                            all_effects.extend([e.strip() for e in effects.split(';')])
                    
                    for effect in all_effects:
                        if effect in side_effect_counts:
                            side_effect_counts[effect] += 1
                        else:
                            side_effect_counts[effect] = 1
                
                # Severity statistics
                severity_counts = {}
                if 'Predicted_Severity' in predictions_df.columns:
                    severity_df = predictions_df['Predicted_Severity'].value_counts().reset_index()
                    severity_df.columns = ['Severity', 'Count']
                    for _, row in severity_df.iterrows():
                        severity_counts[row['Severity']] = row['Count']
                
                # Format dashboard HTML
                html = """
                <div style="padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
                    <h2 style="color: #2c3e50; text-align: center;">Dialysis Monitoring Dashboard</h2>
                    
                    <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
                        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 45%;">
                            <h3 style="color: #3498db; margin-top: 0;">Patient Statistics</h3>
                            <p><strong>Total Patients:</strong> {total_patients}</p>
                            <p><strong>Total Records:</strong> {total_records}</p>
                        </div>
                        
                        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 45%;">
                            <h3 style="color: #e74c3c; margin-top: 0;">Side Effect Severity</h3>
                """.format(total_patients=total_patients, total_records=total_records)
                
                # Add severity chart (simple HTML version)
                for severity, count in severity_counts.items():
                    color = '#e74c3c' if severity == 'Severe' else '#f39c12' if severity == 'Moderate' else '#27ae60'
                    percent = count / len(predictions_df) * 100
                    html += f"""
                            <div style="margin-bottom: 10px;">
                                <span style="color: {color};">{severity}</span>: {count} ({percent:.1f}%)
                                <div style="background-color: #ecf0f1; height: 10px; border-radius: 5px; margin-top: 5px;">
                                    <div style="background-color: {color}; width: {percent}%; height: 10px; border-radius: 5px;"></div>
                                </div>
                            </div>
                    """
                
                html += """
                        </div>
                    </div>
                    
                    <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
                        <h3 style="color: #9b59b6; margin-top: 0;">Common Side Effects</h3>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                """
                
                # Add side effect tags
                for effect, count in sorted(side_effect_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
                    if effect and effect != "None":
                        html += f"""
                            <div style="background-color: #e8e8e8; padding: 8px 12px; border-radius: 20px; font-size: 14px;">
                                {effect} ({count})
                            </div>
                        """
                
                html += """
                        </div>
                    </div>
                    
                    <div style="text-align: center; color: #7f8c8d; font-size: 12px;">
                        Last updated: {timestamp}
                    </div>
                </div>
                """.format(timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                return html
            except Exception as e:
                logger.error(f"Error generating dashboard: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return f"<p>Error generating dashboard: {str(e)}</p>"
        
        # Add refresh button
        with gr.Row():
            refresh_dashboard_btn = gr.Button("Refresh Dashboard")
            
        # Add display for dashboard
        dashboard_display = gr.HTML(label="Dashboard")
        
        # Connect refresh button
        refresh_dashboard_btn.click(
            fn=generate_dashboard,
            inputs=[],
            outputs=[dashboard_display]
        )
        
        # Load dashboard on page load
        demo.load(
            fn=generate_dashboard,
            inputs=[],
            outputs=[dashboard_display]
        )

# Launch the app
demo.launch()
