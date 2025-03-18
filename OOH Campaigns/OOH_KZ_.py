import trino
# Import pacakges and files for SC

#import numpy as np
import pandas as pd
from glovo_synthetic_control_experiments.synthetic_control import SyntheticControl
import plotly.express as px
from matplotlib import style
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

style.use("fivethirtyeight")

# In case you run the code in a JupyterLab cloud environment:
# Else, if running it locally, import data using the trino package
%load_ext starburst