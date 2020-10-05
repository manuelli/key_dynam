import os
import torch


from key_dynam.utils.utils import load_yaml
from key_dynam.dynamics.models_dy import DynaNetMLP, VisualDynamicsNet, DynaNetMLPWeighted, DynaNetMLPWeightMatrix
from key_dynam.dense_correspondence.descriptor_net import PrecomputedDescriptorNet

def build_visual_dynamics_model(config):
    vision_net = None
    dyna_net = None

    vision_net_type = config['vision_net']['model_type']
    if vision_net_type == "PrecomputedDescriptorNet":
        vision_net = PrecomputedDescriptorNet(config)
        vision_net.initialize_weights()
    else:
        raise ValueError("unsupported vision net type")

    dyna_net_type = config['dynamics_net']["model_type"]
    if dyna_net_type == "mlp":
        dyna_net = DynaNetMLP(config)
    else:
        raise ValueError("unsupported dynamics net type")

    visual_dynamics_net = VisualDynamicsNet(config, vision_net, dyna_net)

    return visual_dynamics_net

def build_dynamics_model(config):
    dyna_net = None

    dyna_net_type = config['dynamics_net']["model_type"]
    if dyna_net_type == "mlp":
        dyna_net = DynaNetMLP(config)
    elif dyna_net_type == "mlp_weighted":
        dyna_net = DynaNetMLPWeighted(config)
    elif dyna_net_type == "mlp_weight_matrix":
        dyna_net = DynaNetMLPWeightMatrix(config)
    else:
        raise ValueError("unsupported dynamics net type")

    return dyna_net


def load_dynamics_model_from_folder(model_folder,
                                    state_dict_file=None,
                                    strict=True):
    """
    Builds model and loads parameters using the 'load_state_dict'
    function
    """
    config = load_yaml(os.path.join(model_folder, 'config.yaml'))
    model = build_dynamics_model(config)

    if state_dict_file is None:
        state_dict_file = os.path.join(model_folder, 'net_best_dy_state_dict.pth')

    model.load_state_dict(torch.load(state_dict_file), strict=strict)
    model = model.eval()
    model = model.cuda()
    _, model_name = os.path.split(model_folder)

    return {'model_dy': model,
            'model_name': model_name,
            'config': config}


