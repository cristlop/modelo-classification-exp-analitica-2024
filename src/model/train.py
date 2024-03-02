import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import os
import wandb
import matplotlib
matplotlib.use("agg")  # Configuración del backend para gráficos de Matplotlib
import matplotlib.pyplot as plt

# Cargar datos
wbcd = load_breast_cancer()
feature_names = wbcd.feature_names
labels = wbcd.target_names

X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2)

# Entrenar modelo y obtener predicciones
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva

# Configuración del dispositivo (GPU o CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return TensorDataset(x, y)

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")

def evaluate(model, test_loader):
    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)
    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=32):
    model.eval()
    loader = DataLoader(testing_set, 1, shuffle=False)
    losses = None
    predictions = None
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            
            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
            else:
                losses = torch.cat((losses, loss.view((1, 1))), 0)
                predictions = torch.cat((predictions, pred), 0)

    argsort_loss = torch.argsort(losses, dim=0, descending=True)
    highest_k_losses = losses[argsort_loss[:k]]
    hardest_k_examples = testing_set[argsort_loss[:k]][0]
    true_labels = testing_set[argsort_loss[:k]][1]
    predicted_labels = predictions[argsort_loss[:k]]

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels

def train_and_log(config, experiment_id='99'):
    with wandb.init(
        project="MLOps-mod-classification-2024", 
        name=f"Train Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", 
        job_type="train-model", config=config) as run:
        config = wandb.config
        data = wandb.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()

        training_dataset =  read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
        
        model_artifact = run.use_artifact("linear:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model_linear.pth")
        model_config = model_artifact.metadata
        config.update(model_config)

        model = Classifier(**model_config)  # Asegúrate de tener la clase Classifier definida
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
 
        train(model, train_loader, validation_loader, config)

        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained NN model",
            metadata=dict(model_config))

        torch.save(model.state_dict(), "trained_model.pth")
        model_artifact.add_file("trained_model.pth")
        wandb.save("trained_model.pth")

        run.log_artifact(model_artifact)

    return model

def evaluate_and_log(experiment_id='99', config=None, model=None, X_test=None, y_test=None):
    with wandb.init(project="MLOps-mod-classification-2024", name=f"Eval Model ExecId-{args.IdExecution} Experiment-{experiment_id}"):
        data = wandb.use_artifact('mnist-preprocess:latest')
        data_dir = data.download()

        testing_dataset = read(data_dir, "testing")
        test_loader = DataLoader(testing_dataset, batch_size=config.batch_size)

        # Evaluar el modelo
        loss, accuracy, highest_losses, hardest_examples, true_labels, predictions = evaluate(model, test_loader)

        # Visualizaciones en Weights & Biases después de la evaluación
        fpr, tpr, _ = roc_curve(y_test, y_probas)
        roc_auc = auc(fpr, tpr)
        wandb.log({"roc_auc": roc_auc, "roc_curve": wandb.plot.roc_curve(y_test, y_probas)})
        
        precision, recall, _ = precision_recall_curve(y_test, y_probas)
        average_precision = average_precision_score(y_test, y_probas)
        wandb.log({"average_precision": average_precision, "precision_recall_curve": wandb.plot.precision_recall_curve(y_test, y_probas)})

        # Matriz de confusión
        cm = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()

        # Registrar la imagen en WandB
        wandb.log({"confusion_matrix": wandb.Image(plt)})

        wandb.log({"test/loss": loss, "test/accuracy": accuracy})

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions
