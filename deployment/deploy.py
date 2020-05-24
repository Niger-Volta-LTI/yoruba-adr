import json
import os
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import json_serializer, json_deserializer
from utils import create_dirs
from env import Env
import logging.config

VERSION = 0.1

env = Env()

BASE_DIR = os.getcwd()  # project root
APP_DIR = os.path.dirname(__file__)  # app root
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

create_dirs(LOGS_DIR)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r") as fp:
        json_obj = json.load(fp)
    return json_obj


log_config = load_json(
    filepath=os.path.join('/Users/owahab/Desktop/personal/niger-volta/yoruba-adr/deployment/logging.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger('logger')


def update_endpoint_if_exists():
    return env.model_exists()


def delete_endpoint_and_config():
    """
    Need to manually delete the endpoint and config because of
    https://github.com/aws/sagemaker-python-sdk/issues/101#issuecomment-607376320.
    """
    env.client().delete_endpoint(EndpointName=env.get('model_name'))
    env.client().delete_endpoint_config(EndpointConfigName=env.get('model_name'))


def deploy():
    logger.info(f"Deploying model_name={env.get('model_name')}")

    model_name = env.get('model_name')
    role = sagemaker.get_execution_role()
    pytorch_model = PyTorchModel(
        model_data=env.get('model_data_path'),
        name=model_name,
        framework_version='1.5.0',
        role=role,
        source_dir="../src",
        entry_point='translate.py')

    if env.model_exists():
        delete_endpoint_and_config()

    predictor = pytorch_model.deploy(
        instance_type="ml.m4.xlarge",
        initial_instance_count=1)
    predictor.content_type = 'application/json'
    predictor.serializer = json_serializer
    predictor.deserializer = json_deserializer


if __name__ == '__main__':
    deploy()
