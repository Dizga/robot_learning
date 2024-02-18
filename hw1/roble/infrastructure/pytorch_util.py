from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
    
def build_mlp(
        input_size: int,
        output_size: int,
        **kwargs
    ):
    """
    Builds a feedforward neural network

    arguments:
    n_layers: number of hidden layers
    size: dimension of each hidden layer
    activation: activation of each hidden layer
    input_size: size of the input layer
    output_size: size of the output layer
    output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    try:
        params = kwargs["params"]
    except:
        params = kwargs

    if isinstance(params["output_activation"], str):
        output_activation = _str_to_activation[params["output_activation"]]
    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    layers = kwargs["params"]["layer_sizes"]
    activation = [_str_to_activation[act] for act in kwargs["params"]["activations"]]

    # Initialize the list of layers
    mlp_layers = []

    # Add the first layer (input to first hidden layer)
    mlp_layers.append(nn.Linear(input_size, layers[0]))
    mlp_layers.append(activation[0])

    # Loop over the remaining hidden layers
    for i in range(1, len(layers)):
        # Add the hidden layer
        mlp_layers.append(nn.Linear(layers[i-1], layers[i]))
        # mlp_layers.append(nn.BatchNorm1d(layers[i]))
        mlp_layers.append(activation[i])

    # Add the output layer
    mlp_layers.append(nn.Linear(layers[-1], output_size))
    mlp_layers.append(output_activation)

    # Create the sequential model
    return nn.Sequential(*mlp_layers)

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
