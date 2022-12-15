import os
import azureml.core
from azureml.core import Workspace, Experiment, Dataset, RunConfiguration, Environment
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.pipeline.core import PublishedPipeline, PipelineEndpoint

# Connect to workspace
ws = Workspace.from_config()

# Get default datastore
default_datastore = ws.get_default_datastore()

# Define default DataPath for training data input and make it configurable via PipelineParameter
data_path = DataPath(datastore=default_datastore, path_on_datastore='training_data/')
datapath_parameter = PipelineParameter(name="training_data_path", default_value=data_path)
datapath_input = (datapath_parameter, DataPathComputeBinding(mode='download'))

# Configure runtime environment for our pipeline using AzureML Environment
runconfig = RunConfiguration()
runconfig.environment = Environment.get(workspace=ws, name='training-env')

train_step = PythonScriptStep(name="train-step",
                        source_directory="./",
                        script_name='train.py',
                        arguments=['--data-path', datapath_input],
                        inputs=[datapath_input],
                        runconfig=runconfig,
                        compute_target='cpu-cluster',
                        allow_reuse=False)

steps = [train_step]

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=steps)
pipeline.validate()

# Publish pipeline to AzureML
published_pipeline = pipeline.publish('prepare-training-pipeline-datapath')

# Publish pipeline to PipelineEndpoint (optional, but recommended when using the pipeline with Azure Data Factory)
endpoint_name = 'training-pipeline-endpoint'
try:
    print(f'Pipeline Endpoint with name {endpoint_name} already exists, will add pipeline to it')
    pipeline_endpoint = PipelineEndpoint.get(workspace=ws, name=endpoint_name)
    pipeline_endpoint.add_default(published_pipeline)
except Exception:
    print(f'Will create Pipeline Endpoint with name {endpoint_name}')
    pipeline_endpoint = PipelineEndpoint.publish(workspace=ws,
                                                name=endpoint_name,
                                                pipeline=published_pipeline,
                                                description="New Training Pipeline Endpoint")