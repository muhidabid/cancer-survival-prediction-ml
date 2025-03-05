# **Cancer Survival Prediction Models Comparison**

## Video & Report
Please find the video presentation and comprehensive report for this project [here](https://drive.google.com/drive/folders/17pVnsFQami9umhXfk5zsocYHksiOBD98?usp=drive_link).

## **Overview**
This project compares the performance of different machine learning models—Linear Regression, Gaussian Process Regression (GPR), and Neural Networks—in predicting the survival days of cancer patients. The goal is to assess which model provides the most accurate predictions for patient survival, based on various features in the dataset.

## **Project Description**
We use a dataset from the [Cancer Genomic Data (GDC)](https://portal.gdc.cancer.gov/analysis_page?app=CDave), selecting specific columns related to patient demographics and clinical information. The dataset contains survival data and features relevant to cancer prognosis. We are using the json_to_csv.ipynb to convert the json file to clinical_data_extracted.csv. We then preprocess the data, handle missing values, and then apply three different models: Linear Regression, GPR, and Neural Networks, to predict the number of survival days for each patient.

### **Models Implemented**
1. **Linear Regression**: A simple regression model used to predict the continuous survival days.
2. **Gaussian Process Regression (GPR)**: A probabilistic, non-linear model used to model the survival days with more flexibility.
3. **Neural Network (NN)**: A Multi layer perceptron using torch, to predict survival days by training a multi-layer neural network.

## **Dataset**
The dataset used in this project is from the [GDC portal](https://portal.gdc.cancer.gov/analysis_page?app=CDave). We select a subset of columns that contain relevant clinical and patient data, such as age, gender, prior malignancy, cancer stage, etc.

You can download the dataset by visiting the link above, navigate to the clinical report download JSON file.

## **Dependencies**
This project is built using the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `gpytorch`
- `mpl_toolkits`
- `itertools`

To install the required libraries, run:

```bash
pip install -r requirements.txt
```

## **Results**

The performance of each model is evaluated using the following metrics:

    Mean Absolute Error (MAE)

    Mean Squared Error (MSE)

    R² Score

The results are visualized in the notebooks - lr_gpr.ipynb and neural_network.ipynb.

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
