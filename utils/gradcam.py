import torch
import torch.nn as nn
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

# Define class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to find last conv layer
def get_last_conv_layer(model):
    last_conv = [None]
    
    def find(module):
        for child in module.children():
            if isinstance(child, torch.nn.Conv2d):
                last_conv[0] = child
            find(child)
    
    find(model)
    if last_conv[0] is None:
        raise ValueError("No Conv2d layer found in model")
    return last_conv[0]

# Function to apply Grad-CAM
def apply_gradcam(model, input_tensor, target_layer, title="Model"):
    # Ensure model can compute gradients
    model.train()
    cam_extractor = GradCAM(model, target_layer=target_layer)
    output = model(input_tensor)
    predicted_class = output.argmax().item()
    activation_map = cam_extractor(predicted_class, output)[0].cpu().squeeze().numpy()
    
    # Convert tensors to PIL images
    input_img = to_pil_image(input_tensor.squeeze(0).cpu())
    cam_pil = Image.fromarray(activation_map)
    result = overlay_mask(input_img, cam_pil, alpha=0.5)
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(input_img)
    ax[0].set_title("Input MRI")
    ax[0].axis('off')
    ax[1].imshow(result)
    ax[1].set_title(f"{title}\nPredicted: {class_names[predicted_class]}")
    ax[1].axis('off')
    plt.tight_layout()
    
    # Convert figure to PIL image
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    pil_image = Image.open(buf)
    
    # Restore eval mode
    model.eval()
    
    return pil_image, class_names[predicted_class]