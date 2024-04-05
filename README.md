# Heart Disease Prediction Model Using Logistic Regression

This Python code implements a logistic regression model to predict the likelihood of heart disease based on various features. It uses the scikit-learn library for data preprocessing, model training, and evaluation.

## Data

The code uses the `heart_disease_data.csv` dataset, which contains the following features:

- `age`: Age of the patient
- `sex`: Sex of the patient (0 = female, 1 = male)
- `cp`: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
- `trestbps`: Resting blood pressure (in mm Hg)
- `chol`: Serum cholesterol (in mg/dl)
- `fbs`: Fasting blood sugar (0 = < 120 mg/dl, 1 = > 120 mg/dl)
- `restecg`: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (0 = no, 1 = yes)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
- `ca`: Number of major vessels colored by fluoroscopy
- `thal`: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)
- `target`: Target variable (0 = no heart disease, 1 = heart disease)

## Preprocessing

The code separates the features (`x`) and the target variable (`y`) from the dataset. It then splits the data into training and testing sets using `train_test_split` from `sklearn.model_selection`.

## Model Training

The logistic regression model is initialized using `LogisticRegression` from `sklearn.linear_model`. The model is trained on the training data using the `.fit()` method.

## Model Evaluation

The trained model's accuracy is evaluated on the test data using the `.predict()` method and `accuracy_score` from `sklearn.metrics`.

## Prediction System

The code provides an example of how to use the trained model for prediction. It takes a new set of input data and passes it to the `.predict()` method to obtain the prediction (0 or 1). Based on the prediction, it prints either "You are Healthy" or "Consult your Doctor".

## Usage

1. Ensure you have the required Python libraries installed (`numpy`, `pandas`, `scikit-learn`).
2. Download the `heart_disease_data.csv` dataset and place it in the same directory as the code.
3. Run the Python script.

Note: The code includes a `ConvergenceWarning` from scikit-learn, indicating that the logistic regression model did not converge within the maximum number of iterations. This warning can be addressed by increasing the number of iterations or scaling the data as suggested in the warning message.
