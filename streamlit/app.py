import streamlit as st
import pickle
import pandas as pd
from predict import Train
import requests 
 
# Loading in the model 
with open('./model/RandomForest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('./model/encoding.pkl', 'rb') as columns_file:
    train_columns = pickle.load(columns_file)

def prediction(house_data):
    model = Train()  

    # Create a DataFrame from the user input 
    df = pd.DataFrame([house_data])  

    # Encoding categorical features 
    df = model.encoding(df)

    # Reindex columns to match the training data, adding missing columns with 0s
    df = df.reindex(columns=train_columns, fill_value=0)

    prediction = loaded_model.predict(df)
    return prediction[0]  

def main():
    st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🏡 Price My Home</h1>
    <h4 style='text-align: center; color: #333;'>Unlock the True Value of Your Property with Our House Price Predictor for Belgian Homes</h3>
    """, 
    unsafe_allow_html=True)
    # unsafe_allow_html=True flag use HTML for styling in the Streamlit app
    st.info("Discover the estimated market price of your property based on key features and real-time data.")


    property_type = st.radio('Type of property', ['HOUSE', 'APARTMENT'])
    subproperty_type = st.selectbox('Subtype of property', ['HOUSE', 'APARTMENT', 'DUPLEX', 'VILLA', 
                'EXCEPTIONAL PROPERTY', 'FLAT STUDIO', 'GROUND_FLOOR', 'PENTHOUSE', 'FARMHOUSE',
                'APARTMENT_BLOCK', 'COUNTRY COTTAGE', 'TOWN HOUSE', 'SERVICE_FLAT', 'MANSION',
                'MIXED USE BUILDING', 'MANOR HOUSE', 'LOFT', 'BUNGALOW', 'KOT', 'CASTLE',
                'CHALET', 'OTHER PROPERTY', 'TRIPLEX'])

    region = st.radio('Region', ['FLANDERS', 'BRUSSELS-CAPITAL', 'WALLONIA'])
    province = st.selectbox('Province', ['ANTWERP', 'EAST FLANDERS', 'BRUSSELS', 'WALLOON BRABANT',
    'FLEMISH BRABANT', 'LIÈGE', 'WEST FLANDERS', 'HAINAUT', 'LUXEMBOURG', 'LIMBURG', 'NAMUR'])
    locality = st.text_input('Locality')

    # Converting field into int and float
    zip_code = st.text_input('Postal Code')
    zip_code = int(zip_code) if zip_code else 0

    latitude = st.text_input('Latitude')
    latitude = float(latitude) if latitude else 0.0

    longitude = st.text_input('Longitude')
    longitude = float(longitude) if longitude else 0.0

    
    construction_year = st.text_input('Construction Year')
    construction_year = int(construction_year) if construction_year else 0

    total_area_sqm = st.text_input('Total Area (sqm)')
    total_area_sqm = float(total_area_sqm) if total_area_sqm else 0.0

    nbr_bedrooms = st.slider('Number of Bedrooms', 0, 10)

    equipped_kitchen = st.selectbox('Kitchen Type', ['INSTALLED', 'HYPER EQUIPPED', 'SEMI EQUIPPED', 
        'USA INSTALLED', 'USA HYPER EQUIPPED', 'NOT INSTALLED', 'USA SEMI EQUIPPED', 'USA UNINSTALLED'])
    
    terrace_sqm = st.text_input('Terrace Area (sqm)')
    terrace_sqm = float(terrace_sqm) if terrace_sqm else None  # if empty

    garden_sqm = st.text_input('Garden Area (sqm)')
    garden_sqm = float(garden_sqm) if garden_sqm else None

    state_building = st.selectbox('State of Building', ['TO RENOVATE', 'GOOD', 'AS NEW', 
        'JUST RENOVATED', 'TO BE DONE UP', 'TO RESTORE'])
    
    primary_energy_consumption_sqm = st.text_input('Primary Energy Consumption (kWh/m²)')
    primary_energy_consumption_sqm = float(primary_energy_consumption_sqm) if primary_energy_consumption_sqm else 0.0

    heating_type = st.selectbox('Type of Heating', ['GAS', 'FUELOIL', 'PELLET', 'ELECTRIC', 'SOLAR', 'WOOD', 'CARBON'])
    other_amenities = st.text_input('Choose other amenities: Add 1 for each amenity selected (Swimming Pool, Furnished, Double Glazing, Open Fire), otherwise enter 0.')
    other_amenities = int(other_amenities) if other_amenities else 0

    if st.button('Predict Price'):
        # dictionary represnts data provided by user and feature that will be sent to the model for prediction.
        house_data = {
            'property_type': property_type,
            'subproperty_type': subproperty_type,
            'region': region,
            'province': province,
            'locality': locality,
            'zip_code': zip_code,
            'latitude': latitude,
            'longitude': longitude,
            'construction_year': construction_year,
            'total_area_sqm': total_area_sqm,
            'nbr_bedrooms': nbr_bedrooms,
            'equipped_kitchen': equipped_kitchen,
            'terrace_sqm': terrace_sqm,
            'garden_sqm': garden_sqm,
            'state_building': state_building,
            'primary_energy_consumption_sqm': primary_energy_consumption_sqm,
            'heating_type': heating_type,
            'other_amenities': other_amenities
        }

    
        # result = prediction(house_data)
        # st.success(f'The estimated price is: €{result:,.2f}')
        #function sends a POST request to the API, which is hosted at the URL and send data in JSON.
        response = requests.post('https://immo-eliza-vma8.onrender.com/predict', json=house_data)
        
        #API receives this data, processes it, and returns a response containing the predicted house price
        if response.status_code == 200:
            result = response.json()
            st.write(f"The estimated price is: €{result['price']:,.2f}")
        else:
            st.error(f"Error {response.status_code}: Unable to get prediction from the model.")
            st.write(response.json())  

if __name__ == "__main__":
    main()
