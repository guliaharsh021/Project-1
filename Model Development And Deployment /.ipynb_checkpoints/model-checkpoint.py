import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def build_model(df, location):
    location_df = df[df['City/Locality'] == location]
    
    # Ensure there are enough samples to split
    if len(location_df) < 5:  # Adjust this number as needed
        print(f"Not enough data for location: {location}. Skipping.")
        return None, None

    location_df = location_df[['BHK', 'Property Size (sqft)', 'Furnishing', 'Price (INR)']].reset_index(drop=True)

    encoded_df = pd.get_dummies(location_df, columns=['Furnishing'])
    X = encoded_df.drop(columns=['Price (INR)'])
    y = encoded_df['Price (INR)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train.columns

# Load data
flats_df = pd.read_csv('/Users/guliaharsh021/Downloads/DA Documents /Projects/Project 1/Data Prepration, Processing and Analysis/Data Exploration and Cleaning/Cleaned Data/flats_data_cleaned.csv')
flats_df = flats_df[['City/Locality', 'BHK', 'Property Size (sqft)', 'Furnishing', 'Price (INR)']].reset_index(drop=True)

flats_df = flats_df[flats_df['BHK'] >= 4]

# Build and save model for each location
locations = flats_df['City/Locality'].unique()
models = {}

for location in locations:
    model, columns = build_model(flats_df, location)
    if model is not None:
        models[location] = (model, columns)

with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)