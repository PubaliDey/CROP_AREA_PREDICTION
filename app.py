# from flask import Flask, request, render_template
# import numpy as np
# import pandas as pd
# import sklearn
# import pickle


# # Deserialize the RandomForestClassifier model
# with open('rfc_model.pkl', 'rb') as file:
#     rfc_deserialized = pickle.load(file)

# # Deserialize the scaler
# with open('scaler.pkl', 'rb') as file:
#     scaler_deserialized = pickle.load(file)

# # Deserialize the soil type label encoder
# with open('label_encoder_soil.pkl', 'rb') as file:
#     label_encoder_soil_deserialized = pickle.load(file)

# # Deserialize the district label encoder
# with open('label_encoder_district.pkl', 'rb') as file:
#     label_encoder_district_deserialized = pickle.load(file)

# app = Flask(__name__)

# @app.route('/', methods=['GET'])
# def home():
#     # Render the HTML form on GET request
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract features from the submitted form
#     if request.method == 'POST':
#         Soil_Type = request.form['Soil_Type']
#         N = float(request.form['N'])
#         P = float(request.form['P'])
#         K = float(request.form['K'])
#         pH = float(request.form['pH'])
#         Humidity = float(request.form['Humidity'])
#         Temperature = float(request.form['Temperature'])
#         Rainfall = float(request.form['Rainfall'])
    
#      soil_type_encoded = label_encoder_soil_deserialized.transform([Soil_Type])[0]
    
#     # Preparing the feature DataFrame with the correct order and column names, including trailing spaces
#     features_df = pd.DataFrame([[soil_type_encoded, N, P, K, pH, Humidity, Temperature, Rainfall]],
#                                columns=['soil_type_encoded', 'N ', 'P ', 'K ', 'pH', 'Humidity', 'Temperature ', 'Rainfall '])
    
#     # Scaling the features using the deserialized scaler
#     features_scaled = scaler_deserialized.transform(features_df)
    
#     # Making predictions with the deserialized RandomForestClassifier model
#     district_encoded = rfc_deserialized.predict(features_scaled)
    
#     # Decoding the predicted 'district' using the deserialized label encoder
#     predicted_district = label_encoder_district_deserialized.inverse_transform([district_encoded])[0]
#     predicted_district = predict_district('Alluvial & Red/Yellow', 120, 72, 180, 6, 80, 25, 2200)
# print(f'The predicted district is: {predicted_district}')

    
#     # Instead of returning jsonify, render a template with the prediction
#     return render_template('index.html', result=result)
    

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify, flash, redirect
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model and preprocessors
with open('rfc_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_soil.pkl', 'rb') as file:
    label_encoder_soil = pickle.load(file)

with open('label_encoder_district.pkl', 'rb') as file:
    label_encoder_district = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Extract data from form submission
            Soil_Type = request.form['Soil_Type']
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            pH = float(request.form['pH'])
            Humidity = float(request.form['Humidity'])
            Temperature = float(request.form['Temperature'])
            Rainfall = float(request.form['Rainfall'])

            # Preprocess
            soil_type_encoded = label_encoder_soil.transform([Soil_Type])[0]
            features = np.array([[soil_type_encoded, N, P, K, pH, Humidity, Temperature, Rainfall]])
            features_scaled = scaler.transform(features)

            # Predict
            district_encoded = model.predict(features_scaled)
            predicted_district = label_encoder_district.inverse_transform(district_encoded)[0]

            return render_template('result.html', prediction=predicted_district)
        except Exception as e:
            # If error, redirect back to home and flash a message
            flash('Please ensure all fields are filled correctly.')
            return redirect('/')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
