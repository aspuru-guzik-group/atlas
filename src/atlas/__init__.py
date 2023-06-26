#!/usr/bin/env python

import os
from glob import glob

atlas_home = os.path.dirname(os.path.abspath(__file__))
atlas_scratch = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".scratch"
)
atlas_datasets = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets"
)
__home__ = os.environ.get("ATLAS_HOME") or atlas_home
__scratch__ = os.environ.get("ATLAS_SCRATCH") or atlas_scratch
__datasets__ = os.environ.get("ATLAS_DATASETS") or atlas_datasets


from .utils.logger import MessageLogger

Logger = MessageLogger()
