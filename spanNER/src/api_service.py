import json
from operator import sub
from telnetlib import Telnet
import pandas as pd
import ast
import os
import requests

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any, List

import torch

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from run_ner import predict
import common.utils as util

app = FastAPI()

initialize(config_path="../config", job_name="app")
cfg = compose(config_name="config")
print(OmegaConf.to_yaml(cfg))

util.config_to_abs_paths(cfg, 'entity_classes_json')
util.config_to_abs_paths(cfg.longformer, 'config', 'model', 'tokenizer', 'autotokenizer')
util.config_to_abs_paths(cfg, 'ner_trained_model_path')

configs = cfg

print(configs)

class Data(BaseModel):
    text: str

class BulkData(BaseModel):
    data: List[Data]

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/single_inference")
def single_inference(request: Data):

    data = request.dict()
    data = [data]

    print("Running inference on: ", data)

    dict_list = predict(configs, data)

    return dict_list[0]

@app.post("/bulk_inference")
def bulk_inference(request: BulkData):
    data = request.dict()
    data_list = data['data']
    print("Performing bulk inference on: ", data_list)
    dict_list = predict(configs, data)

    return {"inference": dict_list}
