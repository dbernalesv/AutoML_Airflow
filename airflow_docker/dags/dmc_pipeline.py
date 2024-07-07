import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from pycaret.classification import *
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import subprocess

import warnings
warnings.filterwarnings('ignore')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 6),
    'email ':['diegobernales3@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'AutoML_workflow_demo',
    default_args=default_args,
    description='Pipeline para AutoML y Sumbit a Kaggle',
    schedule_interval='0 17 * * *', # para configurar el horario de ejecucion , se usa crontab
)

def GetDataKaggle():
    api = KaggleApi()
    api.authenticate()
    # Download the competition files
    competition_name = 'playground-series-s4e6'
    download_path = '/opt/airflow/dags/data/'
    api.competition_download_files(competition_name, path=download_path)
    # Unzip the downloaded files
    for item in os.listdir(download_path):
        if item.endswith('.zip'):
            zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
            zip_ref.extractall(download_path)
            zip_ref.close()
            print(f"Unzipped {item}")

def AutoML_PyCaret():

    # Configuración del experimento
    df_train = pd.read_csv('/opt/airflow/dags/data/train.csv').drop(columns=['id'])

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
    
    # metricas
    tuned_better_results = exp_pc01.pull()

    AUC = tuned_better_results.loc[['Mean'],['AUC']].values[0][0]
    F1 = tuned_better_results.loc[['Mean'],['F1']].values[0][0]

    print('AUC: ', AUC, 'F1: ',F1 )
        
    # Finalizar el modelo
    final_model = finalize_model(tuned_model_better)

    # Guardar modelo 
    save_model(final_model, '/opt/airflow/dags/model/modelo_final_pc01')

    #cargar dataset de test
    df_test = pd.read_csv('/opt/airflow/dags/data/test.csv')

    # Realizar predicciones
    predictions = predict_model(final_model, data=df_test)

    print('Se realizaron las predicciones correctamente')

    # Create a DataFrame with 'id' and prediction_label
    result = pd.DataFrame({
        'id': df_test['id'],
        'Target': predictions['prediction_label']
        })

    # Save the result to a CSV file
    predicitions_path = '/opt/airflow/dags/data/submission_0.csv'

    result.to_csv(predicitions_path, index=False)

    return predicitions_path

def SubmitKaggle(ti):

    api = KaggleApi()
    api.authenticate()

    #Subir archivo a Kaggle

    file_name=ti.xcom_pull(task_ids='AutoML_PyCaret')

    if file_name:
        command = [
            'kaggle', 'competitions', 'submit',
            '-c', 'playground-series-s4e6',
            '-f', file_name,
            '-m', 'submission_final'
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar el comando: {e.stderr}")
    else:
        print("No se pudo obtener el nombre del archivo de xcom.")

    #!kaggle competitions submit -c playground-series-s4e6 -f file_name -m "submission_final"

    #api.competition_submit(file_name=ti.xcom_pull(task_ids='AutoML_PyCaret'),
    #                   message="First submission",
    #                   competition="playground-series-s4e6")
    
#######################################################################################################
    
GetDataKaggle_task = PythonOperator(

    task_id='GetDataKaggle',
    python_callable=GetDataKaggle,
    dag=dag,
)

AutoML_PyCaret_task = PythonOperator(
    task_id='AutoML_PyCaret',
    python_callable=AutoML_PyCaret,
    dag=dag,
)

SubmitKaggle_task = PythonOperator(
    task_id='SubmitKaggle',
    python_callable=SubmitKaggle,
    dag=dag,
)

GetDataKaggle_task >> AutoML_PyCaret_task  >> SubmitKaggle_task