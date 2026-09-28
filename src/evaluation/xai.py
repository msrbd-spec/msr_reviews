import cv2
import numpy as np
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    """
    Wrapper to make the dict-returning model compatible with pytorch-grad-cam,
    which expects a tensor output from the forward pass.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out['logits']
        return out


def get_target_layer(model):
    """
    Dynamically finds the appropriate target layer for CAM methods.
    Checks model.named_modules() to ensure we target the exact nested attribute.
    If MSDA is used, we target the final MSDA block's deform_conv.
    Otherwise, we target the final Swin-V2 stage block's norm layer.
    """
    target_layer = None

    # 1. Try to find the final MSDA block if MSDA is enabled
    for name, module in model.named_modules():
        if name == 'msda4.deform_conv':
            target_layer = module
            break

    if target_layer is not None:
        return [target_layer]

    # 2. If MSDA is not enabled, find the final block of the Swin-V2 backbone
    last_norm_name = None
    last_norm_module = None
    for name, module in model.named_modules():
        if 'backbone.layers.3.blocks' in name and 'norm2' in name:
            last_norm_name = name
            last_norm_module = module

    if last_norm_module is not None:
        return [last_norm_module]

    raise ValueError("Could not find a suitable target layer for XAI. Please inspect model.named_modules().")


def generate_heatmap(image_tensor, model, original_image, method='eigencam',
                     out_name='xai_heatmap.png', return_heatmap=False):
    """
    Generates and saves the attention heatmap.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W).
        model (nn.Module): The trained RetiNA_Net model.
        original_image (np.ndarray): Original image in RGB (H, W, 3), values in [0, 1].
        method (str): 'eigencam' or 'gradcam' (case-insensitive).
        out_name (str): Path to save the visualization. If None, doesn't save.
        return_heatmap (bool): If True, returns the raw grayscale heatmap (H, W)
                               instead of the overlay visualization.

    Returns:
        If return_heatmap: np.ndarray (H, W) — raw grayscale heatmap
        Otherwise: np.ndarray (H, W, 3) — overlay visualization
    """
    model.eval()

    # Wrap model so it returns a tensor (required by pytorch-grad-cam)
    wrapped_model = ModelWrapper(model)

    target_layers = get_target_layer(model)

    # Case-insensitive method matching
    method_lower = method.lower()
    if method_lower == 'eigencam':
        cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
    else:
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    # Generate the heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)

    # grayscale_cam has shape (batch_size, H, W)
    grayscale_cam = grayscale_cam[0, :]

    # If only raw heatmap is requested, return it
    if return_heatmap:
        return grayscale_cam

    # Resize original_image if it doesn't match the CAM size
    if original_image.shape[:2] != grayscale_cam.shape:
        original_image = cv2.resize(original_image, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

    # Overlay the cam on the image
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    # Save the visualization
    if out_name is not None:
        cv2.imwrite(out_name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return visualization

