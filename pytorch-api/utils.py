from app import model, imagenet_class_index
import io

import torchvision.transforms as transforms
from PIL import Image

import torch

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    
    m = torch.nn.Softmax(dim=1)
    outputs2 = m(outputs)

    score, y_hat = outputs2.max(1)
    score = score.item()

    predicted_idx = str(y_hat.item())
    return score, imagenet_class_index[predicted_idx]