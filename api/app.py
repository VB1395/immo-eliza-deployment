from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from api.predict import Train  
from typing import Optional, Literal

app = FastAPI()

# Input data model for predictions
class HouseDetails(BaseModel):
    property_type: Literal["House", "Apartment"]
    subproperty_type: str
    region: str
    province: str
    locality: str
    zip_code: int
    latitude: float
    longitude: float
    construction_year: int
    total_area_sqm: float
    nbr_bedrooms: int
    equipped_kitchen: str
    terrace_sqm: Optional[float] 
    garden_sqm: Optional[float]
    state_building: str 
    primary_energy_consumption_sqm: float
    heating_type: str
    other_amenities: Optional [int] 

@app.post("/predict")
def predict(data: HouseDetails):
    try: 
        # Load the trained model model\RandomForest_model.pkl
        with open('./model/RandomForest_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Load the column names used during training
        with open('./model/encoding.pkl', 'rb') as columns_file:
            train_columns = pickle.load(columns_file)

        # Preprocess the input data
        model = Train()  # Create an instance of the Train class for preprocessing
        df = pd.DataFrame([data.model_dump()])  # Convert input data to dictionary

        # Encoding categorical features as  per the training data
        df = model.encoding(df)
        
        # Impute missing values as done during training
        df = model.imputing(df)

        # Reindex columns to match the training data, adding missing columns with 0s
        df = df.reindex(columns=train_columns, fill_value=0)

        # Predict using the trained model
        prediction = loaded_model.predict(df)

        return {"predicted_price": round(prediction[0], 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")


# Main entry point for running the app (for development purposes)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
