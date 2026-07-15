from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flood_prediction'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flood_prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Load the scaler and model
scaler = joblib.load('transform.save')
model = joblib.load('floods.save')

# Database Models
class User(db.Model):
    __tablename__ = 'Users'
    UserID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100), nullable=False)
    Email = db.Column(db.String(100), unique=True, nullable=False)
    Password = db.Column(db.String(200), nullable=False)
    Role = db.Column(db.String(50), default='User')
    weather_data = db.relationship('WeatherData', backref='user', lazy=True)

class MLModel(db.Model):
    __tablename__ = 'ML_Model'
    ModelID = db.Column(db.Integer, primary_key=True)
    ModelName = db.Column(db.String(100))
    AlgorithmType = db.Column(db.String(100))
    Accuracy = db.Column(db.Float)
    ModelFile = db.Column(db.String(100))
    predictions = db.relationship('PredictionResult', backref='model', lazy=True)

class WeatherData(db.Model):
    __tablename__ = 'Weather_Data'
    DataID = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.Integer, db.ForeignKey('Users.UserID'), nullable=False)
    AnnualRainfall = db.Column(db.Float, nullable=False)
    CloudVisibility = db.Column(db.Float, nullable=False)
    Temperature = db.Column(db.Float, nullable=False)
    Humidity = db.Column(db.Float, nullable=False)
    SeasonalRainfall = db.Column(db.Float, nullable=False)
    prediction = db.relationship('PredictionResult', backref='weather', uselist=False)

class PredictionResult(db.Model):
    __tablename__ = 'Prediction_Result'
    PredictionID = db.Column(db.Integer, primary_key=True)
    DataID = db.Column(db.Integer, db.ForeignKey('Weather_Data.DataID'), nullable=False)
    ModelID = db.Column(db.Integer, db.ForeignKey('ML_Model.ModelID'), nullable=False)
    FloodResult = db.Column(db.String(50))
    FloodProbability = db.Column(db.Float)
    PredictionDate = db.Column(db.DateTime, default=datetime.utcnow)

# Create Database and default Model entry
with app.app_context():
    db.create_all()
    if not MLModel.query.first():
        default_model = MLModel(ModelName='XGBoost_Flood', AlgorithmType='XGBoost', Accuracy=81.60, ModelFile='floods.save')
        db.session.add(default_model)
        db.session.commit()

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(Email=email).first()
        if user:
            flash('Email already exists. Please login.')
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(Name=name, Email=email, Password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(Email=email).first()
        if user and check_password_hash(user.Password, password):
            session['user_id'] = user.UserID
            session['user_name'] = user.Name
            return redirect(url_for('predict_page'))
        else:
            flash('Invalid credentials. Please try again.')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for('home'))

@app.route('/Predict', methods=['GET', 'POST'])
def predict_page():
    if 'user_id' not in session:
        flash('Please login to use the prediction tool.')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        # Get inputs
        annual_rainfall = float(request.form['annual_rainfall'])
        cloud_visibility = float(request.form['cloud_visibility'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        seasonal_rainfall = float(request.form['seasonal_rainfall'])
        
        # Save to Weather_Data
        weather = WeatherData(
            UserID=session['user_id'],
            AnnualRainfall=annual_rainfall,
            CloudVisibility=cloud_visibility,
            Temperature=temperature,
            Humidity=humidity,
            SeasonalRainfall=seasonal_rainfall
        )
        db.session.add(weather)
        db.session.commit()
        
        # Predict
        input_data = pd.DataFrame({
            'AnnualRainfall': [annual_rainfall],
            'CloudVisibility': [cloud_visibility],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'SeasonalRainfall': [seasonal_rainfall]
        })
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100
        
        result_text = "High Chance" if prediction == 1 else "Low Chance"
        
        # Save Prediction
        active_model = MLModel.query.first()
        pred_record = PredictionResult(
            DataID=weather.DataID,
            ModelID=active_model.ModelID,
            FloodResult=result_text,
            FloodProbability=probability
        )
        db.session.add(pred_record)
        db.session.commit()
        
        if prediction == 1:
            return redirect(url_for('chance', prob=round(probability, 2)))
        else:
            return redirect(url_for('no_chance', prob=round(probability, 2)))
            
    return render_template('index.html')

@app.route('/chance')
def chance():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    prob = request.args.get('prob', 'High')
    return render_template('chance.html', prob=prob)

@app.route('/no_chance')
def no_chance():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    prob = request.args.get('prob', 'Low')
    return render_template('no_chance.html', prob=prob)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    predictions = db.session.query(PredictionResult, WeatherData).join(
        WeatherData, PredictionResult.DataID == WeatherData.DataID
    ).filter(WeatherData.UserID == session['user_id']).order_by(PredictionResult.PredictionDate.desc()).all()
    
    return render_template('history.html', history=predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
