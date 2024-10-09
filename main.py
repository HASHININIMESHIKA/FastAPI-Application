from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model and label encoders
model_data = joblib.load("salary_prediction_model.pkl")
regressor = model_data["model"]
le_country = model_data["le_country"]
le_education = model_data["le_education"]

# Create the FastAPI app
app = FastAPI()

# Define the input data model
class SalaryInput(BaseModel):
    country: str
    education_level: str
    years_of_experience: float

# Create an endpoint for salary prediction
@app.post("/predict_salary/")
def predict_salary(data: SalaryInput):
    try:
        # Convert the input into the required format
        input_data = np.array([[data.country, data.education_level, data.years_of_experience]])

        # Transform categorical data using label encoders
        input_data[:, 0] = le_country.transform([data.country])
        input_data[:, 1] = le_education.transform([data.education_level])
        input_data = input_data.astype(float)

        # Predict the salary using the loaded model
        predicted_salary = regressor.predict(input_data)

        # Return the prediction 
        return {"predicted_salary": f"${predicted_salary[0]:,.2f}"}
    except Exception as e:
        return {"error": str(e)}

# Add a root endpoint for GET requests
@app.get("/")
def read_root():
    return {"message": "Welcome to the Salary Prediction API. Use the /predict_salary/ endpoint for predictions."}
