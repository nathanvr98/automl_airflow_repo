import json
from pendulum import datetime
import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import logging
from airflow.decorators import (
    dag,
    task,
)
import pandas as pd
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Fill the .env file with your Kaggle API keys

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

if not KAGGLE_USERNAME or not KAGGLE_KEY:
    logging.error("KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment.")
else:
    logging.info("Kaggle credentials loaded successfully.")

class MLSystem:

    def loaddata(self):
        
        df = pd.read_csv('data/train.csv')
        print(df.head())
        return df
    
    def ejecucionmodelo(self, df):
        # Configuración del experimento
        exp_clf101 = setup(data=df, 
                        target='Target', 
                        session_id=123, 
                        train_size=0.7, 
                        ignore_features=['id'], 
                        numeric_features=["Previous qualification","Previous qualification (grade)","Mother's qualification",
                                            "Father's qualification",'Admission grade','Age at enrollment','Curricular units 1st sem (credited)',
                                            'Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)',
                                            'Curricular units 1st sem (approved)','Curricular units 1st sem (grade)',
                                            'Curricular units 1st sem (without evaluations)','Curricular units 2nd sem (credited)',
                                            'Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)',
                                            'Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)',
                                            'Curricular units 2nd sem (without evaluations)','Unemployment rate','Inflation rate','GDP'],
                        categorical_features=['Marital status', 'Application mode', 'Course','Daytime/evening attendance','Nacionality',
                                                "Mother's occupation","Father's occupation",'Displaced', 'Educational special needs', 
                                                'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder','International'])
        
        # Comparación de modelos
        best_model = compare_models()

        # Creación de un modelo específico
        dt = create_model('lightgbm')  # Light Gradient Boosting Machine

        # Optimización de hiperparámetros
        # Define the parameter grid for Grid Search
        param_grid_bayesian = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }

        # Perform Bayesian Search
        tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-learn', search_algorithm='grid')

        return tuned_dt
    
    def evaluarmodelo(self, tuned_dt):
        # Finalizar el modelo
        final_dt = finalize_model(tuned_dt)
        
        # Lee test
        df_test = pd.read_csv('data/test.csv')

        # Realizar predicciones
        predictions = predict_model(final_dt, data=df_test)

        return final_dt, df_test, predictions

    def guardarresultados(self, final_dt, df_test, predictions):
        # Create a DataFrame with 'id' and 'Exited' probabilities
        result = pd.DataFrame({
            'id': df_test['id'],
            'Target': predictions['prediction_label']
        })

        # Save the result to a CSV file
        result.to_csv('submissions/submission.csv', index=False)

        # Guardar y cargar modelos
        save_model(final_dt, 'final_dt_model')

    def __init__(self):
        print("instanciado")

    def ejecucion(self):
        print("Cargando Data...")
        df = self.loaddata()
        print("Ejecutando modelo...")
        tuned_dt = self.ejecucionmodelo(df)
        print("Evaluando modelo...")
        final_dt, df_test, predictions = self.evaluarmodelo(tuned_dt)
        print("Guardando resultados del modelo...")
        self.guardarresultados(final_dt, df_test, predictions)

@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 2,
    },
    tags=["pycaret"],
)
def pycaret_dag():
    @task(task_id="GetDataKaggle")
    def extract_data_from_kaggle():

        competition_name = "playground-series-s4e6"
        download_path = "data/"

        if not os.path.exists(download_path):
            os.makedirs(download_path)
            logging.info(f"Created directory: {download_path}")

        api = KaggleApi()
        api.authenticate()
        logging.info("Kaggle API authenticated successfully.")

        api.competition_download_files(competition_name, path=download_path)
        logging.info(f"Files downloaded to {download_path}")

        # Unzip the downloaded files
        for item in os.listdir(download_path):
            if item.endswith(".zip"):
                zip_ref = zipfile.ZipFile(os.path.join(download_path, item), "r")
                zip_ref.extractall(download_path)
                zip_ref.close()
                logging.info(f"Unzipped {item}")

                extracted_files = zip_ref.namelist()
                logging.info(f"Extracted files: {extracted_files}")

        # List all files in the directory after extraction
        for root, dirs, files in os.walk(download_path):
            for name in files:
                logging.info(f"File found: {os.path.join(root, name)}")

    @task(task_id="AutoML_PyCaret")
    def auto_ml_pycaret():
        modelo = MLSystem()
        modelo.ejecucion()
        return logging.info("AutoML solution created")

    @task(task_id="SubmitKaggle")
    def submit_to_kaggle():
        api = KaggleApi()
        api.authenticate()
        logging.info("Kaggle API authenticated successfully.")
        api.competition_submit(
            file_name="submissions/submission.csv",
            message="First submission",
            competition="playground-series-s4e6",
        )
        return logging.info("Solution submitted to Kaggle")

    extract_data_task = extract_data_from_kaggle()
    auto_ml_task = auto_ml_pycaret()
    submit_task = submit_to_kaggle()

    extract_data_task >> auto_ml_task >> submit_task

dag_instance = pycaret_dag()
