FROM apache/airflow:2.8.3-python3.11

# Verificar la instalación de Python
RUN python --version

USER root
# Copiar el script de instalación al contenedor
COPY install_os_dependencies.sh /install_os_dependencies.sh

# Ejecutar el script de instalación
RUN chmod +x /install_os_dependencies.sh && /install_os_dependencies.sh

# Copiar archivos necesarios
COPY requirements.txt /requirements.txt
COPY kaggle.json /home/airflow/.kaggle/kaggle.json

# Instalar pip y paquetes de Python
USER airflow
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt