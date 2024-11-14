import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

filename = r"C:\Users\Vinay\Desktop\Becode\Projects\immo-eliza-deployment\data\houses.csv"

class Train:
    
    def load_file(self, filename):
        """Load CSV file"""
        df = pd.read_csv(filename)
        return df
    
    def cleaning(self, df):
        """Clean the dataset"""
        other_amenities_col = ['fl_furnished', 'fl_double_glazing', 'fl_open_fire', 'fl_swimming_pool', 'fl_floodzone']
        df['other_amenities'] = df[other_amenities_col].sum(axis=1)
        
        df.drop(['id', 'nbr_frontages', 'cadastral_income', 'epc', 'surface_land_sqm', 
                 'fl_terrace', 'fl_garden', 'fl_furnished', 'fl_double_glazing', 
                 'fl_open_fire', 'fl_swimming_pool', 'fl_floodzone','latitude', 'longitude','locality'], axis=1, inplace= True)
        
        # Drop rows with missing values in crucial columns
        df = df.dropna(subset=['construction_year', 'primary_energy_consumption_sqm'], axis=0)
        
        return df
    
    def encoding(self, df):
        """Encode categorical columns"""
        categorical_columns = ['property_type', 'region', 'heating_type', 'province', 'subproperty_type']
        
        # Handle missing values by filling NaNs with a placeholder like 'MISSING'
        df[categorical_columns] = df[categorical_columns].fillna('MISSING')
        
        # One-Hot Encoding for categorical variables
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoder_array = encoder.fit_transform(df[categorical_columns])
        encoder_df = pd.DataFrame(encoder_array, columns=encoder.get_feature_names_out())
        df = pd.concat([df.drop(columns=categorical_columns), encoder_df], axis=1)
        
        # Ordinal Encoding for 'equipped_kitchen' column
        encoding_kitchen = OrdinalEncoder(categories=[['MISSING', 'NOT_INSTALLED', 'USA_UNINSTALLED', 'INSTALLED',
        'USA_INSTALLED', 'SEMI_EQUIPPED', 'USA_SEMI_EQUIPPED', 'HYPER_EQUIPPED', 'USA_HYPER_EQUIPPED']],handle_unknown='use_encoded_value', unknown_value=-1)
        df['equipped_kitchen'] = encoding_kitchen.fit_transform(df[['equipped_kitchen']])
        
        # Ordinal Encoding for 'state_building' column
        encoding_state_of_building = OrdinalEncoder(categories=[['MISSING', 'TO_RESTORE', 'TO_RENOVATE',
        'TO_BE_DONE_UP', 'GOOD', 'AS_NEW', 'JUST_RENOVATED']],handle_unknown='use_encoded_value', unknown_value=-1)
        df['state_building'] = encoding_state_of_building.fit_transform(df[['state_building']])
        
        # Update 'construction_year' to the difference with the reference year (2024)
        reference_year = 2024
        df['construction_year'] = reference_year - df['construction_year']
        
        return df
    
    def imputing(self, df):
        """Impute missing values"""
        df['total_area_sqm'] = df['total_area_sqm'].fillna(df['total_area_sqm'].median())
        df['terrace_sqm'] = df['terrace_sqm'].fillna(df['terrace_sqm'].median())
        df['garden_sqm'] = df['garden_sqm'].fillna(df['garden_sqm'].median())
        return df
    
    def training(self, df):
        """Train the model"""
        # Remove rows where the target column 'price' is NaN
        df = df.dropna(subset=['price'])
        
        # Split data into features (X) and target (y)
        X = df.drop(['price'], axis=1)
        y = df['price']  # Reshaping target for regression
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        
        
        # Initialize and train the RandomForest model
        model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2)
        model.fit(X_train, y_train)   
        
        print('Trainig Score:',model.score(X_train,y_train))
        print('Testing Score:',model.score(X_test,y_test))
        y_pred = model.predict(X_test)
    
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
        # Print the metrics
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        with open('model/RandomForest_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        
        
        
        # Save the column names used after encoding
        with open('model/encoding.pkl', 'wb') as columns_file:
            pickle.dump(X_train.columns, columns_file)

        print("Model file, and encoded columns saved....")


# Main script execution
if __name__ == "__main__":
    train = Train()
    df = train.load_file(filename)  # Load the data
    df = train.cleaning(df)         # Clean the data
    df = train.encoding(df)         # Encode categorical variables
    df = train.imputing(df)         # Impute missing values
    train.training(df)              # Train the model and save it
