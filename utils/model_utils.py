import math

import torch
import torch.nn as nn

from models.layers.moe_layer import MoEConv


def set_scores(model, scores):
    for module in model.modules():
        if isinstance(module, MoEConv):
            module.scores = scores


def clear_scores(model):
    for module in model.modules():
        if isinstance(module, MoEConv):
            module.scores = None


def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def initialize_scores(model, init_type):
    print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )


def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                    math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )


def collect_conv_layer(model):
    conv_name_list = [name for name, module in model.named_modules()
                      if isinstance(module, nn.Conv2d) and "shortcut" not in name]

    return conv_name_list


def set_mode(model, mode):
    assert mode in ["model", "router", "joint"]

    if mode == "model":
        for name, module in model.named_modules():
            if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm2d)):
                if "router" in name:
                    if hasattr(module, "weight"):
                        if getattr(module, "weight") is not None:
                            getattr(module, "weight").requires_grad = False
                    if hasattr(module, "bias"):
                        if getattr(module, "bias") is not None:
                            getattr(module, "bias").requires_grad = False
                else:
                    if hasattr(module, "weight"):
                        if getattr(module, "weight") is not None:
                            getattr(module, "weight").requires_grad = True
                    if hasattr(module, "bias"):
                        if getattr(module, "bias") is not None:
                            getattr(module, "bias").requires_grad = True

    elif mode == "router":
        for name, module in model.named_modules():
            if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm2d)):  # keep bn changeable
                if "router" in name:
                    if hasattr(module, "weight"):
                        if getattr(module, "weight") is not None:
                            getattr(module, "weight").requires_grad = True
                    if hasattr(module, "bias"):
                        if getattr(module, "bias") is not None:
                            getattr(module, "bias").requires_grad = True
                else:
                    if hasattr(module, "weight"):
                        if getattr(module, "weight") is not None:
                            getattr(module, "weight").requires_grad = False
                    if hasattr(module, "bias"):
                        if getattr(module, "bias") is not None:
                            getattr(module, "bias").requires_grad = False
    else:
        for name, module in model.named_modules():
            if "router" in name:
                if hasattr(module, "weight"):
                    if getattr(module, "weight") is not None:
                        getattr(module, "weight").requires_grad = True
                if hasattr(module, "bias"):
                    if getattr(module, "bias") is not None:
                        getattr(module, "bias").requires_grad = True
            else:
                if hasattr(module, "weight"):
                    if getattr(module, "weight") is not None:
                        getattr(module, "weight").requires_grad = True
                if hasattr(module, "bias"):
                    if getattr(module, "bias") is not None:
                        getattr(module, "bias").requires_grad = True


def show_gradients(model):
    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")
        pass


def get_score_gradient_function(model, score_gradient):
    grad_scalar_list = []
    ind = 0
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                getattr(v, "popup_scores").grad.retain_graph = True
                # print(getattr(v, "popup_scores").grad.grad_fn)
                grad_scalar_list.append(torch.sum(getattr(v, "popup_scores").grad * score_gradient[ind]))
                ind += 1
    grad_scalar = torch.tensor(sum(grad_scalar_list), requires_grad=True)
    return grad_scalar


def get_param(model):
    grad_list = []
    for i, v in model.named_modules():
        if hasattr(v, "weight"):
            if getattr(v, "weight") is not None:
                grad_list.append(getattr(v, "weight"))
        if hasattr(v, "bias"):
            if getattr(v, "bias") is not None:
                grad_list.append(getattr(v, "bias"))
    return grad_list


# def initialize_scaled_score(model):
#     print(
#         "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
#     )
#     sum_parameter = 0
#     for m in model.modules():
#         if hasattr(m, "popup_scores"):
#             n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
#             # Close to kaiming unifrom init
#             m.popup_scores.data = (
#                     math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
#             )
#             # print(f"m's parameter count {torch.sum(m.popup_scores.data != 0)}")
#             sum_parameter += torch.sum(m.popup_scores.data != 0)
#     # print(f"parameter count {sum_parameter}")


def scale_rand_init(model, k):
    print(
        f"Initializating random weight with scaling by 1/sqrt({k}) | Only applied to CONV & FC layers"
    )
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data = 1 / math.sqrt(k) * m.weight.data


def copy_router_grad_and_clear(model):
    grad = {}
    for name, param in model.named_parameters():
        if "router" in name:
            grad[name] = param.grad.clone()
            param.grad = None
    return grad


def set_router_grad_and_clear(model, grad):
    for name, param in model.named_parameters():
        if "router" in name:
            param.grad = grad[name]
        else:
            param.grad = None


def analyze_router_res_distribution(router_res, expert_num, logger):
    for i, res in enumerate(router_res):
        logger.info(f"For {i}-th MoE layer:")
        stat = []
        total = len(res)
        for expert_id in range(expert_num):
            stat.append(((res == expert_id).sum() / total * 100))
            # logger.info(f"The load of {expert_id} router is {(res==expert_id).sum() / total * 100: .2f}")
        logger.info(f"ROUTER ANALYSIS: {i}-th MoE layer: {stat}")
