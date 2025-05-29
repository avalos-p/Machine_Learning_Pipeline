# Machine Learning Pipeline for Credit Card Fraud Detection

This project implements an end-to-end ETL Machine Learning pipeline for detecting credit card fraud. The pipeline is orchestrated using Apache Airflow, containerized with Docker, and built in Python. This project serves as a practical demonstration of building and deploying a machine learning system.

## Features

*   **Automated ETL Process:** Uses Apache Airflow to define, schedule, and monitor the data pipeline.
*   **Data Extraction:** Fetches data from a PostgreSQL database using a SQL query.
*   **Data Cleaning & Preprocessing:** Includes steps for cleaning raw data (`cleaner.py`).
*   **Feature Engineering:**
    *   **Categorical Encoding:** Transforms categorical features into a numerical format using Ordinal Encoding (`encoder.py`). Allows for saving the trained encoder or using a pre-existing one.
    *   **Numerical Scaling:** Scales numerical features using RobustScaler to handle outliers effectively (`robust_scaler.py`). Allows for saving the trained scaler or using a pre-existing one.
*   **Advanced Model Training:**
    *   Uses a `DecisionTreeClassifier` as the base model (`model2.py`).
    *   Employs Scikit-learn `Pipeline` to chain preprocessing and modeling steps.
    *   Performs hyperparameter optimization using `GridSearchCV` to find the best model configuration.
    *   Handles imbalanced datasets by optimizing for metrics like PR AUC.
*   **Model Evaluation:** Evaluates the trained model using a comprehensive set of metrics suitable for imbalanced classification (Precision, Recall, F1-score, ROC AUC, PR AUC, Confusion Matrix).
*   **Artifact Management:** Saves the trained model (pipeline), evaluation metrics (JSON), and performance plots (ROC curve, Precision-Recall curve).
*   **Configuration Driven:**
    *   Model parameters, hyperparameter grids, and output paths are managed via a JSON configuration file (`dags/src/utils/model_config.json`).
    *   Database credentials and sensitive information are managed using environment variables on Airflow.
    *   Logging is configured using a YAML file (`dags/config/log_config.yml`) to improve Airflow logs.
*   **Containerized Environment:** Utilizes Docker and `docker-compose.yaml` to set up a reproducible environment with Airflow.

## Technologies Used

*   **Orchestration:** Apache Airflow
*   **Containerization:** Docker, Docker Compose
*   **Programming Language:** Python
*   **Data Handling & ML:** Pandas, NumPy, Scikit-learn
*   **Database Interaction:** SQLAlchemy (for `db_helpers.py`).
*   **Plotting:** Matplotlib
*   **Configuration:** JSON, YAML
*   **Environment Management:** `airflow-variables`

## Setup and Installation

1.  **Prerequisites:**
    *   Docker: [Install Docker](https://docs.docker.com/get-docker/)

2.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd Machine_Learning_Pipeline
    ```

3.  **Configure Environment Variables:**
    *   The pipeline expects database credentials. These are loaded via `Variable.get()` in Airflow.
    *   Create the following Airflow Variables (key-value pairs):
        *   `HOST`: Your database host.
        *   `PORT`: Your database port.
        *   `DATABASE`: Your database name.
        *   `USER`: Your database username.
        *   `PASSWORD`: Your database password.

4.  **Build and Run Airflow Environment:**
    *   From the project root directory (where `docker-compose.yaml` is located):
        ```bash
        docker-compose build
        docker-compose up
        ```

## Usage

1.  **Navigate to Airflow UI:** Open `http://localhost:8080`.
2.  **Locate the DAG:** Find the `fraud_detection_etl_and_training_pipeline` DAG in the list.
3.  **Unpause the DAG:** Click the toggle switch to unpause it.
4.  **Trigger the DAG:**
    *   Click on the DAG name.
    *   Click the "Play" button.
5.  **Monitor Execution:** Observe the pipeline tasks progressing through the Airflow UI.
6.  **View Results:**
    *   **Trained Model:** Saved in the directory `dags/models/trained_models/first_model.joblib`.
    *   **Metrics:** Saved in the directory  `dags/models/trained_models/first_model_metrics.json`.
    *   **Plots:** Saved in the directory `dags/plots/`.

## Configuration Details

*   **Pipeline Configuration:** Key parameters for model training, hyperparameter grids, and output paths are defined in `dags/src/utils/model_config.json`.
*   **Logging:** Configured via `dags/config/log_config.yml`. Logs are written to `stdout` and a rotating file handler (e.g., `dags/logs/ML_Pipeline.log`).
*   **Database for Data Source:** The `extract_data_from_db` task connects to a PostgreSQL database. Connection parameters must be set as Airflow Variables (see Setup).
*   **Airflow Backend:** The `docker-compose.yaml` sets up PostgreSQL as the metadata database for Airflow itself.

