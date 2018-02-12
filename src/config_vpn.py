#
# Define dataset, model right here in config for maximum flexibility and DRY.
#

from synthetic_datasets import Ball
from model_vpn import VPN


n_pixvals = 2


# Velocity relates to kernel size
dataset = Ball(
    shape = [25,25],
    radius = [6,6],
    velocity = 1,
    gravity = 0,
    bounce = True
)

model = VPN(
    img_channels = 1,
    c = 46,
    n_rmb_encoder = 4,
    n_rmb_decoder = 4,
    n_context = 2,
    n_pixvals = n_pixvals,
    enc_dilation = [1,2,3,4],
    enc_kernel_size = 3,
    lstm_layers = 1,
    use_lstm_peepholes = True,
    mask = True
)

meta = dict(
    # .0003 in paper
    learning_rate = 0.001,
    n_batches = 2000,
    batch_size = 64,
    inputs_seq_len = 10,
    outputs_seq_len = 10,
    infer_n_batches = 5,

    patience = 50,
    gamma = 0.3,

    # Extra stuff that training routine needs to know
    n_pixvals = n_pixvals
)
