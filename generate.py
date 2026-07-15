import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # Temperature: 20 to 45 degrees Celsius
    temperature = np.random.uniform(20, 45, num_samples)
    
    # Humidity: 30% to 100%
    humidity = np.random.uniform(30, 100, num_samples)
    
    # Cloud Visibility (Cloud Cover): 0 to 100%
    cloud_visibility = np.random.uniform(0, 100, num_samples)
    
    # Annual Rainfall: 500mm to 4000mm
    annual_rainfall = np.random.uniform(500, 4000, num_samples)
    
    # Seasonal Rainfall: 100mm to 2000mm
    seasonal_rainfall = np.random.uniform(100, 2000, num_samples)
    
    # Define logic for flood (class = 1) vs no flood (class = 0)
    # Higher rainfall, humidity, and cloud cover increase the chance
    flood_prob = (
        (annual_rainfall / 4000) * 0.4 +
        (seasonal_rainfall / 2000) * 0.4 +
        (humidity / 100) * 0.1 +
        (cloud_visibility / 100) * 0.1
    )
    
    # Add some random noise
    noise = np.random.normal(0, 0.1, num_samples)
    flood_prob += noise
    
    # Class threshold
    target_class = (flood_prob > 0.65).astype(int)
    
    df = pd.DataFrame({
        'AnnualRainfall': annual_rainfall,
        'CloudVisibility': cloud_visibility,
        'Temperature': temperature,
        'Humidity': humidity,
        'SeasonalRainfall': seasonal_rainfall,
        'class': target_class
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data(5000)
    
    output_path = "flood_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path} with {len(df)} rows.")
    print("Class distribution:")
    print(df['class'].value_counts())
