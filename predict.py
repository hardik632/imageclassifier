import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Image ')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
parser.add_argument('--topk', type=int, help='top K')
parser.add_argument('--category_names', type=str, help='JSON file')
parser.add_argument('--gpu', action='store_true', help='gpu')

args, _ = parser.parse_known_args()

def predict(image, checkpoint, topk=5, labels='', gpu=False):
    if args.image:
        image = args.image        
    if args.checkpoint:
        checkpoint = args.checkpoint
    if args.topk:
        topk = args.topk        
    if args.labels:
        labels = args.category_names
    if args.gpu:
        gpu = args.gpu
    checkpoint_dict = torch.load(checkpoint)
    arch = checkpoint_dict['arch']
    num_labels = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
        
    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

    if gpu and torch.cuda.is_available():
        model.cuda()
        
    training = model.training    
    model.eval()
    
    img_pil = Image.open(image)
   
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(img_pil)
    
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    
    if gpu and torch.cuda.is_available():
        image = image.cuda()
    output = model(image).topk(topk)

    if gpu and torch.cuda.is_available():
        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:       
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = output[1].data.numpy()[0]


    if labels:
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)

        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]
        
    model.train(mode=training)
    if args.image:
        print('Predictions and probs', list(zip(classes, probs)))
    
    return probs, classes

if args.image and args.checkpoint:
    predict(args.image, args.checkpoint)