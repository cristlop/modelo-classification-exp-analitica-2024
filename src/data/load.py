import torch
import torchvision
from torch.utils.data import TensorDataset
from sklearn import datasets as sk_datasets
import argparse
import wandb

# Parsear argumentos de la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID de la ejecución')
args = parser.parse_args()

# Verificar si se proporciona IdExecution
if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    """
    Cargar el conjunto de datos de cáncer de mama, para dividirlo en datasets de entrenamiento y validación.
    """
      
    # Carga dataset de cáncer de mama desde scikit-learn
    wbcd = sk_datasets.load_breast_cancer() 
    feature_names = wbcd.feature_names
    labels = wbcd.target_names

    # Convierte datos a tensores de PyTorch
    x, y = torch.tensor(wbcd.data).float(), torch.tensor(wbcd.target)

    # Divide el dataset en conjuntos de entrenamiento y validación
    split_idx = int(len(x) * train_size)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Crea objetos TensorDataset para conjuntos de entrenamiento y validación
    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)

    datasets = [training_set, validation_set]
    return datasets

def load_and_log():
    """
    Cargar el conjunto de datos de cáncer de mama, registrar información sobre el conjunto de datos 
    y guardarlo como un Artifact de Weights & Biases.
    """
    # Inicia ejecución con Weights & Biases
    with wandb.init(
        project="MLOps-mod-classification-2024",
        name=f"Cargar Datos Crudos ExecId-{args.IdExecution}", job_type="cargar-datos") as run:
        
        # Carga dataset cáncer de mama
        datasets = load()

        # Nombres para los conjuntos de entrenamiento y validación
        names = ["entrenamiento", "validación"]

        # Crea nuevo Artifact de Weights & Biases para el conjunto de datos crudo
        raw_data = wandb.Artifact(
            "wisconsin-cancer-mama-crudo", type="dataset",
            description="conjunto de datos crudo de cáncer de mama de Wisconsin, dividido en tren/val",
            metadata={"fuente": "sklearn.datasets.load_breast_cancer",
                      "tamaños": [len(dataset) for dataset in datasets]})

        # Almacenar datos de entrenamiento y validación en el Artifact
        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # Guardar el Artifact en Weights & Biases
        run.log_artifact(raw_data)

# Prueba la función load_and_log
load_and_log()
