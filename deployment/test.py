import boto3
import json
from env import Env

env = Env()

runtime = boto3.Session().client(service_name='sagemaker-runtime', region_name='eu-central-1')

print("Attempting to invoke model_name=%s / env=%s..." % (env.get('model_name'), env.current_env))

payload = [{
    'src': '../data/test/one_phrase.txt',
    'tgt': '../data/test/one_phrase.target.txt',
    'output': '../data/test/pred.txt',

}]

response = runtime.invoke_endpoint(
    EndpointName=env.get("model_name"),
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps(payload)
)

print("Response=", response)
response_body = json.loads(response['Body'].read())
print(json.dumps(response_body, indent=4))
