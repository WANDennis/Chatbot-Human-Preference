#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss

# %% read data
TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"
SS_CSV = "./data/sample_submission.csv"
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
ss = pd.read_csv(SS_CSV)
# print(test.head(1))

# %%
target_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
train[target_cols].sum().plot(kind='bar', title='Raw Counts of Target in Training Set')
plt.show()

# %%
# eval(train.loc[0]['prompt'])[0]
print(f"==== Model A [{train.loc[0]['model_a']}] ({train.loc[0]['winner_model_a']})==========")
# %%
