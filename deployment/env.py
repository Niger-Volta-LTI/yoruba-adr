import os

import boto3
import botocore
import sagemaker
import yaml


class Env:
    def __init__(self):
        self._client = None
        self._runtime_client = None
        self.config_dirname = os.path.dirname(__file__)
        self.config_filename = os.path.join(self.config_dirname, 'config.yaml')
        self.data = self.load_config()

    def load_config(self):
        with open(self.config_filename, 'r') as stream:
            try:
                return yaml.safe_load(
                    stream.read(),
                )
            except yaml.YAMLError as exc:
                print(exc)

    @property
    def current_env(self):
        return os.environ.get("ENVIRONMENT", "local")

    def get(self, name):
        return self.data["environments"][self.current_env][name]

    def model_exists(self):
        """
        Checks if the model is deployed.
        IMPORTANT: always returns `False` for local endpoints as LocalSagemakerClient.describe_endpoint()
        seems to always throw:
        botocore.exceptions.ClientError: An error occurred (ValidationException) when calling the describe_endpoint operation: Could not find local endpoint
        """
        model_exists = False
        client = self.get_client()
        try:
            client.describe_endpoint(EndpointName=self.get("model_name"))
            model_exists = True
        except botocore.exceptions.ClientError as e:
            pass

        return model_exists

    def runtime_client(self):
        if self.current_env == "local":
            runtime_client = sagemaker.local.LocalSagemakerRuntimeClient()
        else:
            runtime_client = boto3.client('sagemaker-runtime')

        return runtime_client

    def get_client(self):
        if self.current_env == "local":
            client = sagemaker.local.LocalSagemakerClient()
        else:
            client = boto3.client('sagemaker')

        return client