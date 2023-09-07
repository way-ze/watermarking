import os
import json

from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import torch

from bigram_estimator import getWordDict, getNgramDict, pLM
from generate_watermark import load_model, generate
from likelihood_ratio_test import L_Gw