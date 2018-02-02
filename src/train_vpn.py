import os, sys, argparse, json, logging, pickle, glob, importlib
from os.path import abspath, join
from time import time

import numpy as np
import torch
from torch.autograd.variable import Variable

from model_vpn import VPN
from synthetic_datasets import Ball
import matplotlib as mpl

# Set non-graphical backend for utils.plotting
if __name__ == "__main__":
    mpl.use('Agg')
import utils

# Get standard logger
root = logging.getLogger()
root.setLevel(logging.DEBUG)

# Set stdout output
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
vanilla_formatter = logging.Formatter('%(message)s')
ch.setFormatter(vanilla_formatter)
root.addHandler(ch)


def save_progress(save_dir, model, losses=None, learning_rates=None, iteration=None):
    """
    Utility function for saving model, losses, and a plot.
    :param save_dir: Path to directory to save results in
    :param model: Subclass of nn.Module to save (pickle)
    :kwarg losses: Optional array of losses per timestep
    :kwarg learning_rates: Optional array of learning rates (same timesteps as losses)
    :kwarg iteration: Index to use when saving the file
    """
    torch.save(model, join(save_dir, "train_convseq2seq_{i}.pt".format(i=iteration)))
    if losses:
        with open(join(save_dir, 'losses.pkl'), 'wb') as f:
            pickle.dump({'losses': losses, 'learning_rates': learning_rates}, f)


def train(a, save_dir=None, save_every=None, logfile=None, use_cuda=True, multi_gpu=False):
    """
    Train loop for ConvSeq2Seq model.
    Will catch a single ctrl-c KeyboardInterrupt and return results.

    :kwarg save_every: If save_dir is specified, save model every 200 iterations.
    :kwarg save_dir:   Save model and summary plots in this dir (recommended to make new one for every experiment).
    :kwarg logfile:    File to log progress
    :kwarg use_cuda:   True by default
    :kwarg multi_gpu:  Use all available GPUs

    :arg a: A dict with the following options

        # GPU use
        use_cuda              True by default

        # Ball params
        input_size            Input size uniform distribution (low, high)
        radius                Radius uniform distribution     (low, high)
        velocity              Max x/y velocity
        gravity               Gravity
        bounce                Boolean for wall bounce

        # Network params
        input_dim             Input channels (usually 1 for greyscale)
        hidden_dim            List of hidden dimensions for each layer, e.g. [32,16,16] for 3-layer
        kernel_size           Tuple of kernel size, e.g. (5,5)

        # Meta params
        learning_rate         Learning rate, e.g. 0.4
        n_batches             Number of batches (dataset is infinite), e.g. 600
        batch_size            Batch size, depends on GPU memory and model size, adjust learning rate accordingly
        inputs_seq_len        Length of input sequence, e.g. 4
        outputs_seq_len       Length of predicted sequence, e.g. 6, these sum to sequence length

        # Learning rate scheme
        step_size             Alter learning rate every so many steps
        gamma                 Multiply learning rate by this factor every *step_size* steps

    :returns: Tuple of (model, losses, learning_rate)
    """

    # Set to sane default: 10 saves in model run
    save_every = np.ceil(a.n_batches / 10).astype(int)

    # Velocity relates to kernel size
    b = Ball(shape=a.input_size, radius=a.radius, velocity=a.velocity, gravity=a.gravity,
             bounce=a.bounce)

    model = VPN(img_channels=a.input_dim, c=a.hidden_dim,
                n_rmb_encoder=a.n_rmb_encoder, n_rmb_decoder=a.n_rmb_decoder,
                n_context=a.n_context,
                n_pixvals = a.n_pixvals,
                enc_dilation = a.enc_dilation,
                enc_kernel_size = a.enc_kernel_size,
                lstm_layers = a.lstm_layers,
                use_lstm_peepholes = a.use_lstm_peepholes,
                mask = a.mask)

    # Define loss function
    loss_func = torch.nn.BCEWithLogitsLoss()

    # Define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=a.learning_rate)

    # Multiply learning rate by *gamma* every *step_size* steps
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=a.step_size, gamma=a.gamma)

    losses = []
    learning_rates = []

    if use_cuda:
        model.cuda()
        if multi_gpu:
            n_gpus = torch.cuda.device_count()
            logging.info("Trying to use {} available GPUs...".format(n_gpus))
            # Assuming that gpus are numbered sequentially and we can use all of them
            model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))

    if save_dir:
        # Make absolute for easy further handling
        save_dir = abspath(save_dir)

        # Create dir if needed
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Set log file logging output
        ch = logging.FileHandler(logfile)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # Add this filehandler to logger
        root.addHandler(ch)

        # Save model layout and config for later reference
        with open(join(save_dir, "config_summary.txt"), 'w') as f:
            f.write(str(model))
            f.write('\n\n')
            config_summary = '\n'.join(('{} = {}'.format(n, getattr(a, n)) for n in dir(a)))
            f.write(config_summary)

    try:
        for i_b in range(a.n_batches):
            t1 = time()
            # Learning rate scheduler
            scheduler.step()

            # (b, t, c, h, w)
            batch = b(batch_size=a.batch_size, sequence_length=a.inputs_seq_len + a.outputs_seq_len + 1)

            batch_t = torch.from_numpy(batch)
            # OFFSET INPUTS AND TARGETS
            inputs_var = Variable(batch_t[:,:-1], requires_grad=True)
            # TODO: Change to use real data?
            targets_var = Variable(batch_t[:,1:])
            # onehot needs LongTensor
            targets_onehot_var = utils.onehot.onehot(targets_var.data, a.n_pixvals)

            if use_cuda:
                inputs_var = inputs_var.cuda()
                targets_var = targets_var.cuda()
                targets_onehot_var = targets_onehot_var.cuda()

            # Perform inference every so often, measure test error
            # We do this on the current batch before training on the current batch as we're working with infinite data.
            # Change to use test set if using real data.
            # if i_b > 0 and (i_b % 1) == 0:
            #     t_evalstart = time()
            #     model.eval()
            #     inputs_var_volatile = Variable(torch.from_numpy(batch[:a.infer_n_batches,:a.inputs_seq_len]), volatile=True)
            #     if use_cuda:
            #         inputs_var_volatile = inputs_var_volatile.cuda()
            #     preds = model(inputs_var_volatile, n_predict=a.outputs_seq_len)
            #     oh = utils.onehot.onehot(preds.data, a.n_pixvals)
            #     if use_cuda:
            #         oh = oh.cuda()
            #     print(oh.size(), targets_onehot_var[:a.infer_n_batches, a.inputs_seq_len:].size())
            #     loss = loss_func(oh,
            #                      targets_onehot_var[:a.infer_n_batches, a.inputs_seq_len:])
            #     logging.info("Loss on fully predicted seq: {:.5f} min {:.2f} max {:.2f} t_eval={:.5f}s"
            #                  .format(loss.data[0],
            #                          preds.min().data[0], preds.max().data[0],
            #                          time()-t_evalstart))

            t2 = time()

            # Explicitly set to train mode
            model.train()
            # Needs targets to condition decoders on during training
            preds = model(inputs_var, targets=targets_var)

            t3 = time()

            # Don't take loss into account for inputs_seq_len
            loss = loss_func(preds[:,a.inputs_seq_len:], targets_onehot_var[:,a.inputs_seq_len:])

            t4 = time()

            # Don't forget to zero the gradient
            optim.zero_grad()
            # Calc all gradients
            loss.backward()
            # Step optimizer
            optim.step()
            t5 = time()

            # Remember learning rate for later graphing
            lr = scheduler.get_lr()
            learning_rates.append(lr)

            logging.info(("Batch {:4d} loss: {:.5f} min {:.2f} max {:.2f} lr={}"
                          " t_gen={:.2f}s t_fwd={:.2f}s t_loss={:.2f}s t_bwd={:.2f}s b/s={:.2f}")
                         .format(i_b, loss.data[0],
                                 preds.min().data[0], preds.max().data[0],
                                 lr,
                                 t2 - t1, t3 - t2, t4 - t3, t5 - t4, 1 / (t5 - t1)))
            losses.append(loss.data[0])

            # Plot losses often
            if save_dir and (i_b % 10) == 0:
                utils.plotting.plot_loss(losses, learning_rates=learning_rates, save_as=join(save_dir, 'summary.png'))
            # Save model if save location specified
            if save_dir and (i_b % save_every == 0):
                save_progress(save_dir, model, losses, learning_rates, iteration=i_b)



    # Enable user to use ctrl-c to prematurely stop training but still return results
    except KeyboardInterrupt:
        logging.info("Caught KeyboardInterrupt, stopping training...")

    if save_dir:
        # Use length of losses as we can't be sure if i_b is defined in this scope
        save_progress(save_dir, model, losses, learning_rates, iteration=len(losses))

    return model, losses, learning_rates


# Cmdline interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VPN on dataset')

    parser.add_argument('settings_file', metavar='SETTINGS_JSON',
                        help="JSON settings file")
    parser.add_argument('save_dir', metavar='SAVE_DIR',
                        help="Directory to save results")
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false',
                        help="Disable CUDA")
    parser.add_argument('--multi-gpu', action='store_true',
                        help="Use all available GPUs with nn.DataParallel")
    parser.add_argument('--clear', action='store_true',
                        help="Clear result directory")

    args = parser.parse_args()

    if args.clear:
        logging.info("Clearing directory {}...".format(args.save_dir))
        files = glob.glob(join(abspath(args.save_dir), '*'))
        for file in files:
            os.remove(file)
        logging.info("Removed {} files".format(len(files)))

    logfile = join(args.save_dir, 'log.txt')

    # Load settings
    settings = importlib.import_module(args.settings_file)
    # Train
    train(settings, save_dir=args.save_dir, logfile=logfile, use_cuda=args.use_cuda, multi_gpu=args.multi_gpu)
