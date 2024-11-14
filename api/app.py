from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from api.predict import Train  
from typing import Optional

app = FastAPI()

@app.get("/")
def status():
    return {"Status": "Alive"}

# Input data model for predictions
class HouseDetails(BaseModel):
    subproperty_type: str
    region: str
    province: str
    zip_code: int
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
        # loading train model
        with open('./model/RandomForest_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Loading encoding model
        with open('./model/encoding.pkl', 'rb') as columns_file:
            train_columns = pickle.load(columns_file)

        # Preprocess and encode input data
        model = Train()  
        df = pd.DataFrame([data.model_dump()])  # Convert input data to dictionary

        df = model.encoding(df)

        # Reindex columns to match the training data, adding missing columns with 0s
        df = df.reindex(columns=train_columns, fill_value=0)

        prediction = loaded_model.predict(df)

        return {"price": round(prediction[0], 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
