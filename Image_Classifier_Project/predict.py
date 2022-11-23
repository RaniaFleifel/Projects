import argparse
import time
import torch
import numpy as np
import json
import sys
from PIL import Image

def load_checkpoint(filepath):
    
    checkpoint  = torch.load(filepath)
    arch=checkpoint['arch']
    if arch=='densenet':
        ip=1024
    else:
        ip=25088
        
    model=checkpoint['model']
   
    model.classifier=nn.Sequential(nn.Linear(ip,checkpoint['hidden']),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(checkpoint['hidden'],102),
                                   nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpoint['state_dict'])  
    model.parameters(checkpoint['parameters'])
    
    
    return model,checkpoint

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    np_adjusted=[]
    # TODO: Process a PIL image for use in a PyTorch model
        #resize the images where the shortest side is 256 pixels, keeping aspect ratio
    im=Image.open(image)
     
    width,height=im.size
    coord=width,height
    min_dimension=min(coord)
    ratio=width/height
    #print(coord.index(min_dimension))
    
    if (coord.index(min_dimension)==0):
        im_resized = im.resize((256, int(256/ratio)))
    elif (coord.index(min_dimension)==1):
        print(coord.index(min_dimension))
        im_resized = im.resize((int(256*ratio), 256))
        
    #Then crop out the center 224x224 portion of the image
    resized_w,resized_h=im_resized.size
    #print(resized_w,resized_h)
        
    factor=224/2
    left=resized_w-factor
    right=resized_w+factor
    upper=resized_h-factor
    lower=resized_h+factor
        
    #print(left, upper, right, lower)    
        # Here the image "im" is cropped and assigned to new variable im_crop
    im_cropped = im_resized.crop((left, upper, right, lower))
        
        
        #Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1.
        #means=[0.485, 0.456, 0.406] std_dev=[0.229, 0.224, 0.225] , subtract the means then divide by the deviation
        
    np_image=np.array(im_cropped)
    #print(im_cropped)
    np_image = np_image.astype('float64')
    x=[225,225,225]
    np_encoded=np_image/x
        
    means=[0.485, 0.456, 0.406]
    std_devs=[0.229, 0.224, 0.225]
    np_adjusted=(np_encoded-means)/std_devs
    
    #Transpose
    np_adjusted=np_adjusted.transpose((2,0,1))
    return np_adjusted

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    #image=torch.from_numpy(image)
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def classify(image_path, topk=5):
    # TODO: Implement the code to predict the class from an image file
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        #class_to_idx=checkpoint['class_to_index_dict']
        image=process_image(image_path)

        image=torch.from_numpy(image)
        image
        image.unsqueeze_(0)
        
        image = image.float()

        
        model,modelinfo=load_checkpoint()
        log_op=model(image)
        op=torch.exp(log_op)

        probs,classes=op.topk(topk)
        probs=probs[0].tolist()
        #classes=classes[0].tolist()
        #THE missing part 
        #convert from these indices to the actual class labels
        direct_mapping=modelinfo['class_to_index_dict'].items()
        
        idx_to_class=[x for x,y in direct_mapping]
        classes=[idx_to_class[x] for x in classes.tolist()[0]]
            
    
        results=zip(probs,classes)
        return results

# stackoverflow help
def use_json_categories():
    if args.category_names is not None:
        cat_file= args.category_names
        jfile=json.loads(open(cat_file).read())
        return jfile
    return None
 

def view_result(results):
    jfile=use_json_categories
    ptr=0
    for p,c in results:
        ptr=ptr+1
        p=str(round(p,4)*100)
        if(jfile):
            c=jfile.get(str(c),'None')
        else:
            c='class{}'.format(str(c))
        print("{}.{} ({})".format(i,c,p))
    return None
##
def parse():
    
    parser=argparse.ArgumentParser(description='Use NN to classify an image into the most probable class')
    parser.add_argument('--image_input')
    parser.add_argument('--model_checkpoint')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names')
    parser.add_argument('--gpu')
    args=parser.parse_args()
    return args

def main():
    global args
    args =parse()
    if args.top_k is None:
        top_k=5
    else:
        top_k=args.top_k
    image_path=args.image_input
    prediction=classify(image_path,top_k)
    view_result(prediction)
    return prediction

main()
    
    
    