import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.data_preprocessing import SatelliteImageDataset, transform
from src.constants import *
import yaml
import torchvision
from pathlib import Path
from src import logger


class ModelTrainer:
    def __init__(self):
        logger.info("Initializing ModelTrainer...")
        
        # Device Configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device selected: {self.device}")

        # Load Dataset
        logger.info("Loading dataset...")
        dataset = SatelliteImageDataset(config_path=CONFIG_PATH, transform=transform)
        dataset_size = len(dataset)
        train_data_size = int(0.8 * dataset_size)
        val_data_size = dataset_size - train_data_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_data_size, val_data_size])
        logger.info(f"Dataset loaded: {dataset_size} samples (Train: {train_data_size}, Validation: {val_data_size})")

        # Load Parameters
        logger.info("Loading parameters from YAML...")
        with open(PARAM_PATH, "r") as file:
            self.param = yaml.load(file, Loader=yaml.SafeLoader)
        logger.info(f"Parameters loaded: {self.param}")

        logger.info("Loading configurations from YAML...")
        with open(CONFIG_PATH, "r") as file:
            self.config = yaml.load(file, Loader=yaml.SafeLoader)
        logger.info(f"Configurations loaded: {self.config}")


        self.train_loader = DataLoader(self.train_dataset, batch_size=self.param['BATCH_SIZE'], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.param['BATCH_SIZE'], shuffle=False)
        logger.info(f"Dataloader initialized (Batch size: {self.param['BATCH_SIZE']})")

        # Initialize Model
        logger.info("Initializing model...")
        self.num_classes = 4  # Update based on your dataset
        self.model = self._initialize_model().to(self.device)
        logger.info("Model initialized successfully.")

        # Loss, Optimizer, and Scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param['LEARNING_RATE'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        logger.info("Loss function, optimizer, and scheduler set up.")

    def _initialize_model(self):
        """Initialize the ResNet model with a modified fully connected layer."""
        logger.info("Setting up the ResNet model...")
        model = torchvision.models.resnet18(pretrained=True)

        # Fine-tune all layers
        for param in model.parameters():
            param.requires_grad = True

        # Replace the fully connected layer to match the number of classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        logger.info("ResNet model setup complete.")
        return model

    def train(self):
        """Train the model for a specified number of epochs."""
        logger.info("Starting training process...")
        num_epochs = self.param['EPOCHS']
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}...")
            self.model.train()
            running_loss = 0.0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                logger.info(f"Processing batch {batch_idx + 1}/{len(self.train_loader)}...")
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Scheduler step
            self.scheduler.step()
            avg_loss = running_loss / len(self.train_loader)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] completed. Loss: {avg_loss:.4f}")

        # Save the trained model
        logger.info("Saving the trained model...")
        model_dir = Path(self.config['model_training']['root_dir'])  # Example: "artifacts/model_training"
        model_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        model_path = model_dir / "resnet_model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model training completed and saved to {model_path}!")


if __name__ == "__main__":
    logger.info("Starting ModelTrainer...")
    trainer = ModelTrainer()
    trainer.train()
