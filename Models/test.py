# connect to comet
from comet_ml import start
from dotenv import load_dotenv
import os
from comet_ml.integration.pytorch import log_model

from model import CompostLSTM


load_dotenv()

experiment = start(
    api_key=os.getenv("SMART_COMPOST_COMET_API_KEY"),
    project_name=os.getenv("SMART_COMPOST_PROJECT_NAME"),
    workspace=os.getenv("SMART_COMPOST_WORKSPACE"),
)


# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 0.05,
    "steps": 1000,
    "batch_size": 5,
}
experiment.log_parameters(hyper_params)

# Initialize and train your model
model = CompostLSTM()
# train(model)

# Seamlessly log your Pytorch model
log_model(experiment, model=model, model_name="CompostLSTM")
