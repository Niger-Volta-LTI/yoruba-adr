from botocore.exceptions import ClientError
from sagemaker.predictor import json_serializer, json_deserializer
import boto3
from sagemaker.session import Session
from sagemaker.pytorch.model import PyTorchModel
import click


@click.command()
@click.option('--instance_count', default=1, help='Instance count to deploy')
def deploy(instance_count):
    model_data = "s3://model-demo-bucket/model.tar.gz"
    role = "yoruba-adr-deployment"
    print("Deploying model")
    name = f"yoruba-adr-pytorch-instance-{instance_count}"

    session = boto3.session.Session(profile_name='yoruba-adr')
    sagemaker_session = Session(boto_session=session)
    client = session.client('sagemaker')
    try:
        client.describe_endpoint(EndpointName=name)
        print("Endpoint exists. Deleting.")
        client.delete_endpoint(EndpointName=name)
        client.delete_endpoint_config(EndpointConfigName=name)
    except ClientError:
        print("Endpoint does not exist")
    finally:
        print("Deleted old endpoint. Creating new endpoint")
        pytorch_model = PyTorchModel(
            model_data=model_data,
            name=name,
            sagemaker_session=sagemaker_session,
            framework_version='1.5.0',
            role=role,
            entry_point='translate.py')

        predictor = pytorch_model.deploy(
            instance_type="ml.t2.medium",  # Smallest instance type that doesn't raise a size error during deployment.
            # update_endpoint = update_endpoint_if_exists() isn't working so
            # https://github.com/aws/sagemaker-python-sdk/issues/101#issuecomment-607376320 is a work around.
            initial_instance_count=instance_count)
        predictor.content_type = 'application/json'
        predictor.serializer = json_serializer
        predictor.deserializer = json_deserializer


if __name__ == '__main__':
    deploy()
