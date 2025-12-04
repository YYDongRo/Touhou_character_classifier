import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import create_model
from src.dataset import TouhouImageDataset


def load_model():
    ds = TouhouImageDataset("data")
    model = create_model(len(ds.class_to_idx))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model, ds.idx_to_class


def get_gradcam(image_path):
    model, idx_to_class = load_model()
    model.eval()

    target_layer = model.layer4[-1]

    activations = None
    gradients = None

    def fwd_hook(m, i, o):
        nonlocal activations
        activations = o

    def bwd_hook(m, gi, go):
        nonlocal gradients
        gradients = go[0]

    target_layer.register_forward_hook(fwd_hook)
    target_layer.register_backward_hook(bwd_hook)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    out = model(x)
    pred = out.argmax().item()
    pred_label = idx_to_class[pred]

    model.zero_grad()
    out[0, pred].backward()

    act = activations.squeeze(0)
    grad = gradients.squeeze(0)

    weights = grad.mean(dim=(1, 2))

    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = F.relu(cam)

    cam -= cam.min()
    cam /= cam.max()

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(224,224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().numpy()

    return cam, img, pred_label


def generate_cam_overlay(image_path):
    cam, img, label = get_gradcam(image_path)
    return cam, img, label
