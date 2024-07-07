import pandas as pd
from pycaret.classification import *
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

os.environ['KAGGLE_USERNAME'] = 'diegobernales'
os.environ['KAGGLE_KEY'] = '412aa99bf4f9f87db726f5b753412751'

import warnings
warnings.filterwarnings('ignore')

class MLSystem:
    def __init__(self):
        pass

    def GetDataKaggle(self):
        api = KaggleApi()
        api.authenticate()
        # Download the competition files
        competition_name = 'playground-series-s4e6'
        download_path = './data_kaggle/'
        api.competition_download_files(competition_name, path=download_path)
        # Unzip the downloaded files
        for item in os.listdir(download_path):
            if item.endswith('.zip'):
                zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
                zip_ref.extractall(download_path)
                zip_ref.close()
                print(f"Unzipped {item}")

    def load_data(self):
        # Simula la carga de un dataset
        df_train = pd.read_csv('./data_kaggle/train.csv').drop(columns=['id'])
        df_test = pd.read_csv('./data_kaggle/test.csv')

        return df_train, df_test
    
    def AutoML_PyCaret(self, df_train):
        # Configuración del experimento
        exp_pc01 = setup(data= df_train, 
                   target='Target', 
                   session_id=123, 
                   train_size=0.7
                   )
        
        # Comparación de modelos
        best_model = compare_models(sort = 'AUC')

        # Create and tune the best model
        tuned_model_better = tune_model(best_model, n_iter = 50, choose_better = True)

        print(tuned_model_better)

        #plot confusion matrix
        exp_pc01.plot_model(tuned_model_better, 
                plot = 'confusion_matrix', 
                plot_kwargs = {'percent' : True},
                save = True
                )
        
        # Finalizar el modelo
        final_model = finalize_model(tuned_model_better)

        # Guardar modelo 
        save_model(final_model, './modelo_final_pc01')

        return final_model, exp_pc01
    
    def evaluate_model(self, exp_pc01):
        # metricas
        tuned_better_results = exp_pc01.pull()

        AUC = tuned_better_results.loc[['Mean'],['AUC']].values[0][0]
        F1 = tuned_better_results.loc[['Mean'],['F1']].values[0][0]

        return AUC, F1

    def predictions(self, model, df_test):
        # Realizar predicciones
        predictions = predict_model(model, data=df_test)

        # Create a DataFrame with 'id' and prediction_label
        result = pd.DataFrame({
            'id': df_test['id'],
            'Target': predictions['prediction_label']
        })

        # Save the result to a CSV file
        result.to_csv('./submission_0.csv', index=False)


    def run_entire_workflow(self):
        try:
            self.GetDataKaggle()
            df_train, df_test = self.load_data()

            print('Se cargó la data correctamente')

            model, exp_pc01 = self.AutoML_PyCaret(df_train)

            print('Se entrenó el modelo correctamente')

            AUC ,F1 = self.evaluate_model(exp_pc01)

            print('AUC: ', AUC)
            print('F1: ', F1)

            self.predictions(model,df_test)

            print('Se realizaron las predicciones correctamente')

            return {'success': True, 'AUC': AUC, 'F1':F1 , 'message': 'el proceso se ha completado'}

        except Exception as e:
            return {'success': False, 'message': str(e)}