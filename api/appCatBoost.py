from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

app = FastAPI()

class HouseDetails(BaseModel):
    property_type: str
    subproperty_type: str
    region: str
    province: str
    locality: str
    zip_code: int
    latitude: float
    longitude: float
    construction_year: float
    total_area_sqm: float
    nbr_bedrooms: float
    equipped_kitchen: str
    terrace_sqm: float
    garden_sqm: float
    state_building: str
    primary_energy_consumption_sqm: float
    heating_type: str
    other_amenities: int


@app.post("/predict")
def predict(data: HouseDetails):
    # Load the model
    model = pickle.load(open('./model/CatBoost.pkl', 'rb'))

    # Prepare the data for prediction
    input_data = [
        data.property_type,
        data.subproperty_type,
        data.region,
        data.province,
        data.locality,
        data.zip_code,
        data.latitude,
        data.longitude,
        data.construction_year,
        data.total_area_sqm,
        data.nbr_bedrooms,
        data.equipped_kitchen,
        data.terrace_sqm,
        data.garden_sqm,
        data.state_building,
        data.primary_energy_consumption_sqm,
        data.heating_type,
        data.other_amenities
    ]

    # Make the prediction
    make_prediction = model.predict([input_data])
    price = round(make_prediction[0], 2)

    return {"prediction": price, "status_code": 200}

@app.get("/")
def index():
    return {"message": "API is working"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
