import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)
app.secret_key = '3d6f45a5fc12445dbac2f59c3b6c7cb1'  # Replace with a secure secret key

# Paths to save models and preprocessors
MODEL_DIR = 'models'
KNN_MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.joblib')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """
    Loads the coffee dataset.
    """
    data_path = os.path.join('data', 'coffee_data.csv')
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Encodes categorical variables and scales numerical features.
    Returns the feature matrix, encoder, and scaler.
    """
    categorical_features = ['Color', 'Roast Level', 'Flavor Profile', 'Acidity', 'Body', 'Origin']
    numerical_features = ['Caffeine Content (mg)']

    # One-Hot Encoding for categorical features
    encoder = OneHotEncoder(sparse_output=False)
    X_categorical = encoder.fit_transform(df[categorical_features])
    categorical_feature_names = encoder.get_feature_names_out(categorical_features)
    df_categorical = pd.DataFrame(X_categorical, columns=categorical_feature_names)

    # Scaling numerical features
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(df[numerical_features])
    df_numerical = pd.DataFrame(X_numerical, columns=numerical_features)

    # Combine all features
    X = pd.concat([df_categorical, df_numerical], axis=1)
    y = df['Name']  # Target variable

    return X, y, encoder, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')

        if not algorithm:
            flash('Please select an algorithm to train.', 'warning')
            return redirect(url_for('train'))

        df = load_data()
        X, y, encoder, scaler = preprocess_data(df)

        # Save the encoder and scaler
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)

        if algorithm == 'knn':
            model = NearestNeighbors(n_neighbors=5, metric='euclidean')
            model.fit(X)
            joblib.dump(model, KNN_MODEL_PATH)
            flash('K-Nearest Neighbors model trained and saved successfully!', 'success')
        elif algorithm == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, RF_MODEL_PATH)
            flash('Random Forest model trained and saved successfully!', 'success')
        else:
            flash('Invalid algorithm selected.', 'danger')
            return redirect(url_for('train'))

        return redirect(url_for('train'))

    return render_template('train.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')

        # Collect user inputs
        color = request.form.get('color').strip().title()
        roast_level = request.form.get('roast_level').strip().title()
        flavor_profile = request.form.get('flavor_profile').strip().title()
        acidity = request.form.get('acidity').strip().title()
        body = request.form.get('body').strip().title()
        origin = request.form.get('origin').strip().title()
        caffeine = request.form.get('caffeine')
        n_recommendation = request.form.get('n_recommendation')

        # Input validation
        if not all([color, roast_level, flavor_profile, acidity, body, origin, caffeine, algorithm, n_recommendation]):
            flash('Please fill out all fields and select an algorithm.', 'warning')
            return redirect(url_for('recommend'))

        try:
            caffeine = float(caffeine)
        except ValueError:
            flash('Caffeine content must be a number.', 'danger')
            return redirect(url_for('recommend'))

        try:
            n_recommendation = int(n_recommendation)
        except ValueError:
            flash('Number of recommendation content must be an integer number.', 'danger')
            return redirect(url_for('recommend'))
        
        user_preferences = {
            'Color': color,
            'Roast Level': roast_level,
            'Flavor Profile': flavor_profile,
            'Acidity': acidity,
            'Body': body,
            'Origin': origin,
            'Caffeine Content (mg)': caffeine
        }

        # Load encoder and scaler
        try:
            encoder = joblib.load(ENCODER_PATH)
            scaler = joblib.load(SCALER_PATH)
        except:
            flash('Model encoders/scalers not found. Please train the model first.', 'danger')
            return redirect(url_for('recommend'))

        # Preprocess user input
        categorical_features = ['Color', 'Roast Level', 'Flavor Profile', 'Acidity', 'Body', 'Origin']
        numerical_features = ['Caffeine Content (mg)']

        user_df = pd.DataFrame([user_preferences])

        # Encode categorical features
        X_categorical = encoder.transform(user_df[categorical_features])
        categorical_feature_names = encoder.get_feature_names_out(categorical_features)
        df_categorical = pd.DataFrame(X_categorical, columns=categorical_feature_names)

        # Scale numerical features
        X_numerical = scaler.transform(user_df[numerical_features])
        df_numerical = pd.DataFrame(X_numerical, columns=numerical_features)

        # Combine all features
        user_X = pd.concat([df_categorical, df_numerical], axis=1)

        # Ensure all feature columns match the training data
        # Load the model to determine feature alignment
        if algorithm == 'knn':
            if not os.path.exists(KNN_MODEL_PATH):
                flash('KNN model not found. Please train the model first.', 'danger')
                return redirect(url_for('recommend'))
            model = joblib.load(KNN_MODEL_PATH)
            # Find the nearest neighbors
            distances, indices = model.kneighbors(user_X)
            # Load the original dataset
            df = load_data()
            recommended_coffees = df.iloc[indices[0]][['Name', 'Origin', 'Flavor Profile', 'Caffeine Content (mg)']]
            recommended_coffees = recommended_coffees.drop_duplicates().head(n_recommendation)
        elif algorithm == 'rf':
            if not os.path.exists(RF_MODEL_PATH):
                flash('Random Forest model not found. Please train the model first.', 'danger')
                return redirect(url_for('recommend'))
            model = joblib.load(RF_MODEL_PATH)
            # Predict probabilities
            probabilities = model.predict_proba(user_X)[0]
            # Get class names
            class_names = model.classes_
            # Create a DataFrame of classes and their probabilities
            prob_df = pd.DataFrame({
                'Name': class_names,
                'Probability': probabilities
            })
            # Sort by probability in descending order
            prob_df = prob_df.sort_values(by='Probability', ascending=False)
            # Select top N recommendations
            top_coffees = prob_df.head(n_recommendation)['Name'].tolist()
            # Fetch details of the recommended coffees
            df = load_data()
            recommended_coffees = df[df['Name'].isin(top_coffees)][['Name', 'Origin', 'Flavor Profile', 'Caffeine Content (mg)']]
        else:
            flash('Invalid algorithm selected.', 'danger')
            return redirect(url_for('recommend'))

        return render_template('recommend.html', coffees=recommended_coffees.to_dict(orient='records'))

    return render_template('recommend.html')

if __name__ == '__main__':
    app.run(debug=True)
