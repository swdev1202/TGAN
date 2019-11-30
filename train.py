import pandas as pd
from tgan.model import TGANModel

data = pd.read_csv('examples/data/hcv.csv')

continuous_columns = [0,2,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

tgan = TGANModel(
    continuous_columns,
    output='output',
    gpu=None,
    max_epoch=10,
    steps_per_epoch=10000,
    save_checkpoints=True,
    restore_session=True,
    batch_size=200,
    z_dim=200,
    noise=0.2,
    l2norm=0.00001,
    learning_rate=0.001,
    num_gen_rnn=100,
    num_gen_feature=100,
    num_dis_layers=1,
    num_dis_hidden=100,
    optimizer='AdamOptimizer'
)

tgan.fit(data)
model_path = 'examples/demo/my_model'
tgan.save(model_path)
print("train finished")