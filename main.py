from src import logger
from src.data_ingestion import DataIngestion
from src.constants import *
from src.data_preprocessing import SatelliteImageDataset,transform
from src.model_training import ModelTrainer
from src.model_evaluation import Model_Evaluation

config_path = CONFIG_PATH
param_path = PARAM_PATH


STAGE_NAME = "Data Ingestion"
logger.info(f"------------->{STAGE_NAME} Stage Started<-----------------")
data_ingestion = DataIngestion()
data_ingestion.download_dataset()
data_ingestion.extract_zip_dataset()
logger.info(f"------------->{STAGE_NAME} Stage Finished<-----------------")


STAGE_NAME = "Data Preprocessing"
logger.info(f"------------->{STAGE_NAME} Stage Started<-----------------")
dataset = SatelliteImageDataset(config_path=config_path,transform=transform)
logger.info(f"------------->{STAGE_NAME} Stage Finished<-----------------")


STAGE_NAME = "Model Training"
logger.info(f"------------->{STAGE_NAME} Stage Started<-----------------")
trainer = ModelTrainer()
trainer.train()
logger.info(f"------------->{STAGE_NAME} Stage Finished<-----------------")


STAGE_NAME = "Model Evaluation"
logger.info(f"------------->{STAGE_NAME} Stage Started<-----------------")
evaluator = Model_Evaluation()
evaluator.evaluate()
logger.info(f"------------->{STAGE_NAME} Stage Finished<-----------------")


