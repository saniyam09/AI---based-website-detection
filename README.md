# AI-Based Phishing Website Detection Using Machine Learning

This project is a beginner-friendly web application that detects whether a website URL is likely legitimate or phishing using machine learning.

## Project Description

The application accepts a URL from the user, extracts phishing-related features, and passes them to a trained machine learning model. The model classifies the URL as either legitimate or phishing and returns a confidence score.

The project follows a simple workflow:

1. Load and prepare a phishing URL dataset.
2. Extract suspicious URL features.
3. Train multiple classifiers.
4. Compare their performance.
5. Save the best model.
6. Use the model in a Flask web application to analyze new URLs.

## Features

- URL input box for user analysis
- Machine learning-based phishing detection
- Confidence score display
- Feature extraction summary panel
- Model comparison chart
- Prediction history for the current session
- Responsive UI for desktop and mobile
- Beginner-friendly code structure
- Error handling for invalid URLs

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML
- CSS
- JavaScript

## Project Structure

AI-Phishing-Website-Detection/

├── app.py
├── train_model.py
├── feature_extraction.py
├── requirements.txt
├── README.md
├── model/
│   ├── phishing_model.pkl
│   └── model_metadata.json
├── data/
│   └── phishing_dataset.csv
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js

## File Overview

### app.py
Runs the Flask web application and exposes the prediction API.

### train_model.py
Loads the dataset, prepares features, trains multiple models, selects the best one, and saves it to disk.

### feature_extraction.py
Contains shared feature extraction logic used both during training and for live predictions so the model and prediction pipeline remain consistent.

### requirements.txt
Lists all required dependencies.

### templates/index.html
Contains the home page layout and dashboard.

### static/style.css
Styles the application for a modern, responsive design.

### static/script.js
Handles frontend behavior, API calls, chart rendering, and UI updates.

## Installation

1. Clone or download the project folder.
2. Open a terminal in the project directory.
3. Create a virtual environment:

   python -m venv venv

4. Activate the virtual environment:

   On Windows:
   venv\Scripts\activate

   On macOS/Linux:
   source venv/bin/activate

5. Install dependencies:

   pip install -r requirements.txt

## Training the Model

Run the training script:

python train_model.py

This will:

- create a CSV dataset if one does not exist,
- extract URL features,
- train Logistic Regression, Decision Tree, and Random Forest models,
- compare their accuracy,
- save the best model to model/phishing_model.pkl.

## Running the Web App

Start the Flask server:

python app.py

Then open your browser and go to:

http://127.0.0.1:5000/

## How the Machine Learning Model Works

The model learns to distinguish between benign and phishing URLs by using custom URL-based features such as:

- URL length
- Domain length
- Number of dots and hyphens
- Number of special characters
- Presence of @ symbol
- Presence of an IP address
- HTTPS usage
- Number of digits
- Number of subdomains
- Suspicious keywords such as login, verify, account, secure, update, bank, password, and confirm

These features are computed for every URL during training and for any new URL entered by the user. The best-performing model is then used to make predictions.

## Screenshots

Add screenshots here later after running the application locally.

Example:

- Homepage screenshot
- Analysis result screenshot
- Mobile view screenshot

## Future Improvements

- Add a larger real-world phishing dataset
- Include more advanced features from WHOIS and DNS analysis
- Add a browser extension version
- Use XGBoost or deep learning models
- Provide API support for external applications
- Improve explainability for each prediction

## Safety Notice

This project performs offline URL and string-based analysis only. It does not visit, crawl, open, or execute websites automatically. The prediction is an ML-based risk assessment and is not a guarantee of website safety.

## License

This project is intended for learning and educational purposes.
