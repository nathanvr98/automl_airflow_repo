# Import necessary libraries
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model

# Load your dataset
import pandas as pd

class MLSystem:

    def loaddata(self):
        df = pd.read_csv('data/train.csv')
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

        modelo = MLSystem()
        modelo.ejecucion()
