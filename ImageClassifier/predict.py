import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import random
import os
import numpy as np
from tensorflow import Variable, convert_to_tensor
from torch import unsqueeze
from PIL import Image

def loadCheckpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
        
    return model, optimizer, input_size, output_size, epoch

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    
    if image.width > image.height:        
        image.thumbnail((10000000, 256))
    else:
        image.thumbnail((256, 10000000))
    
    # Center crop the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    image = np.array(image)
    image = image / 255
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    image = image.transpose(2, 0, 1)
    
    return image    

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()
    model.cpu()
   
    image = Image.open(image_path)
    
    image = process_image(image) 
    
    image = torch.from_numpy(image).type(torch.FloatTensor) 

    image = image.unsqueeze(0)
    
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    
    top_probs = top_probs.detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    labs = []
    for label in top_labs.numpy()[0]:
        labs.append(idx_to_class[label])

    return top_probs, labs

parser = argparse.ArgumentParser(description="This program predicts flowers' names from their images",usage='''
        needs a saved checkpoint
        python predict.py ( use default image 'flowers/test/7/image_06743.jpg' and root directory for checkpoint)
        python predict.py /path/to/image checkpoint (predict the image in /path/to/image using checkpoint)
        python predict.py --top_k 3 (return top K most likely classes)
        python predict.py --category_names cat_to_name.json (use a mapping of categories to real names)
        python predict.py --gpu (use GPU for inference)''',prog='predict')

parser.add_argument('path_to_image', action="store", nargs='?', default='flowers/test/7/image_06743.jpg', help="path/to/image")

parser.add_argument('path_to_checkpoint', action="store", nargs='?', default='checkpoint.pth', help="path/to/checkpoint")

parser.add_argument('--top_k', action="store", default=1, type=int, help="enter number of guesses", dest="top_k")

parser.add_argument('--category_names', action="store", default="cat_to_name.json", help="get json file", dest="category_names")

parser.add_argument('--gpu', action="store_true", default=False, help="Select GPU", dest="gpu")


args = parser.parse_args()

arg_path_to_image =  args.path_to_image
arg_path_to_checkpoint = args.path_to_checkpoint
arg_top_k =  args.top_k
arg_category_names =  args.category_names

if args.gpu and torch.cuda.is_available(): 
    arg_gpu = args.gpu

elif args.gpu:
    arg_gpu = False
    print('GPU is not available, will use CPU...')
    print()

else:
    arg_gpu = args.gpu


device = torch.device("cuda" if arg_gpu else "cpu")
print()
print('Will use {} for prediction...'.format(device))
print()

print()
print("Path of image: {} \nPath of checkpoint: {} \nTopk: {} \nCategory names: {} ".format(arg_path_to_image, arg_path_to_checkpoint, arg_top_k, arg_category_names))
print('GPU: ', arg_gpu)
print()


print('Mapping from category label to category name...')
print()
with open(arg_category_names, 'r') as f:
    cat_to_name = json.load(f)


print('Loading model')
print()

my_model, my_optimizer, input_size, output_size, epoch  = loadCheckpoint(arg_path_to_checkpoint)

my_model.eval()


idx_to_class = {v:k for k, v in my_model.class_to_idx.items()}



print(arg_path_to_image)
probs, classes = predict('{}'.format(arg_path_to_image), my_model, arg_top_k)

print()
print('The model predicts this flower as: ')
print()
for count in range(arg_top_k):
     print('{} ...........{:.3f} %'.format(cat_to_name[idx_to_class[classes[0, count].item()]], probs[0, count].item()))