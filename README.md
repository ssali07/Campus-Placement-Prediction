# Campus Placement Prediction App

A web-based application built using Flask that predicts the probability of a student getting placed in campus recruitment based on various features like age, gender, branch of study, internships, CGPA, and more.

## Demo
Below are the screenshots of the web app:

### Main Page
[![Main Page UI](main-page-ui.png)](https://github.com/your-username/campus-placement-prediction-app)

### Prediction Page
[![Prediction Page UI](prediction-page-ui.png)](https://github.com/your-username/campus-placement-prediction-app)


## Features
- Input features include:
  - Age
  - Gender (Male/Female)
  - Branch of study (Electronics and Communication, Computer Science, Information Technology, Mechanical, Electrical, Civil)
  - Internship experience
  - CGPA
  - Hostel status
  - History of backlogs
- Displays the probability of getting placed.
- Simple and interactive user interface.
- Backend developed using Flask and the machine learning model implemented with XGBoost.

## Technologies Used
- **Frontend**: HTML (rendered using Flask's Jinja2 templates)
- **Backend**: Python, Flask
- **Machine Learning Model**: XGBoost
- **Model Deployment**: joblib for model and scaler serialization
- **Libraries**: 
  - Numpy for numerical operations
  - Joblib for model persistence
  - XGBoost for the prediction model

## Getting Started
To get a local copy up and running, clone the repository with `git clone https://github.com/ssali07/Campus-Placement-Prediction.git`, navigate into the project directory using `cd Campus-Placement-Prediction`, ensure you have the trained model file (`placement_model.pkl`) and scaler file (`scaler.pkl`) placed in the project directory, install the required Python libraries with `pip install -r requirements.txt`, run the Flask application using `python app.py`, and open your web browser to go to `http://127.0.0.1:5000/` to use the app.


## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

