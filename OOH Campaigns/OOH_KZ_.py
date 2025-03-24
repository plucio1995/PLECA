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

## Starbust credentials (this works alone, no need to modify)
HOST = 'starburst.g8s-data-platform-prod.glovoint.com'
PORT = 443
conn_details = {
    'host': HOST,
    'port': PORT,
    'http_scheme': 'https',
    'auth': trino.auth.OAuth2Authentication()
}