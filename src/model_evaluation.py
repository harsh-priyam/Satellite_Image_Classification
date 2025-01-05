import copy
import torch
from src.constants import *
from src.data_preprocessing import SatelliteImageDataset, transform
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import yaml
import torchvision
from pathlib import Path
from src import logger

class Model_Evaluation:
    def __init__(self):
        # Initialize parameters and load the model
        logger.info("Initializing Model_Evaluation...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device selected: {self.device}")
        self.num_classes = 4  # Ensure this matches the number of classes during training

        logger.info("Loading parameters from YAML...")
        with open(PARAM_PATH, "r") as file:
            self.param = yaml.load(file, Loader=yaml.SafeLoader)
        logger.info(f"Parameters loaded: {self.param}")

        logger.info("Loading configurations from YAML...")
        with open(CONFIG_PATH, "r") as file:
            self.config = yaml.load(file, Loader=yaml.SafeLoader)
        logger.info(f"Configurations loaded: {self.config}")

        # Load Dataset
        logger.info("Loading dataset...")
        dataset = SatelliteImageDataset(config_path=CONFIG_PATH, transform=transform)
        dataset_size = len(dataset)
        train_data_size = int(0.8 * dataset_size)
        val_data_size = dataset_size - train_data_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_data_size, val_data_size])
        logger.info(f"Dataset loaded: {dataset_size} samples (Train: {train_data_size}, Validation: {val_data_size})")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.param['BATCH_SIZE'], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.param['BATCH_SIZE'], shuffle=False)
        logger.info(f"Dataloader initialized (Batch size: {self.param['BATCH_SIZE']})")

        # Load the pre-trained model
        self.model = self._initialize_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        logger.info("Model initialized successfully.")

    def _initialize_model(self):
        """Load the ResNet model and apply the saved weights."""
        logger.info("Setting up the ResNet model...")
        model = torchvision.models.resnet18(pretrained=False)  # Do not use pre-trained weights from torchvision
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)

        # Load the trained model weights
        model_path = Path(self.config['model_training']['root_dir'])
        logger.info(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info("Model weights loaded successfully.")

        return model

    def validate(self):
        """Validate the model on the validation dataset."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                y_pred = self.model(images)
                loss = self.criterion(y_pred, labels)
                val_loss += loss.item()

                _, predicted = torch.max(y_pred, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(self.val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_val_loss, accuracy

    def evaluate(self):
        """Start the evaluation process."""
        logger.info("Starting evaluation process...")
        avg_val_loss, val_accuracy = self.validate()
        logger.info(f"Final Evaluation: Validation Loss = {avg_val_loss:.4f}, Accuracy = {val_accuracy:.2f}%")


if __name__ == "__main__":
    logger.info("Starting Model Evaluation...")
    evaluator = Model_Evaluation()
    evaluator.evaluate()
