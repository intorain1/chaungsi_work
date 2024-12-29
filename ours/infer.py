import math
import os.path

from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

torch.set_grad_enabled(False)

CLASSES = [
    'N/A', 'coin1', 'coin2', 'coin3', 'bult', 'nut'
]

COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, width, height):
    box_coords = box_cxcywh_to_xyxy(out_bbox)
    scale_tensor = torch.Tensor(
        [width, height, width, height]).to(
        torch.cuda.current_device()
    )
    return box_coords * scale_tensor


def plot_results(pil_img, prob, boxes, image_item, save_dir):
    save_img_path = os.path.join(save_dir, image_item)

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'

        txt_id = image_item[:-4]


        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(save_img_path, format='jpeg')
    plt.close("all")


def detect(im, model, transform):
    device = torch.cuda.current_device()
    width = im.size[0]
    height = im.size[1]

    img = transform(im).unsqueeze(0)
    img = img.to(device)

    outputs = model(img)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.25

    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], width, height)
    return probas[keep], bboxes_scaled

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=6)
    state_dict = torch.load(r"C:\Users\57704\Downloads\weights\004.pth") #TODO change to your model path
    model.load_state_dict(state_dict)
    
    model.to(device)
    
    model.eval()
    
    image_file_path =  os.listdir(r"C:\Users\57704\Desktop\tri_pre\data_pre\test\images") #TODO if you are using image folder with multiple images, use this. or you can simply change it to singel image path like line105-108
    save_dir = r"C:\Users\57704\Desktop\tri_pre"
    for image_item in image_file_path:
        image_path = os.path.join(r"C:\Users\57704\Desktop\tri_pre\data_pre\test\images", image_item)
        im = Image.open(image_path)
        scores, boxes = detect(im, model, transform)
        plot_results(im, scores, boxes, image_item, save_dir)
