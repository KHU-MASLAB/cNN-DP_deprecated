import pandas as pd

# Data loading
DatasetPath = "Dataset"
ModelPath = "Models"
TrainData = pd.read_csv(f"{DatasetPath}\\TrainData.csv")
ValidData = TrainData

# Hyperparams
global_epochs = 5000
global_halflife = 500
global_batchsize = 128
global_save_every = 25
global_print_every = 100
global_pin_memory = True
global_num_workers = 2
hlayers_others = 8
neurons_others = 350
hlayers_dp = 8
neurons_dp = 200

# Data column names
t = "x"
y = "y"
yDot = "yDot"
yDDot = "yDDot"
