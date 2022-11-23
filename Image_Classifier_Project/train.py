import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import datasets,transforms,models
from torch import nn,optim



#import data_loading

#import model_related.py
def data_loading():
    input_dir=args.dir
    train_dir=input_dir+'/train'
    valid_dir=input_dir+'/valid'
    test_dir=input_dir+'/test'
    data_dir=[train_dir,valid_dir,test_dir]
    
    return data_dir

def def_dataloaders():
    
    input_dir=args.dir
    train_dir=input_dir+'train'
    valid_dir=input_dir+'valid'
    test_dir=input_dir+'test'
    data_dir=[train_dir,valid_dir,test_dir]
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    test_valid_transforms= transforms.Compose([transforms.Resize(230),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_valid_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=test_valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size=64)
    loaders={'train':trainloader,'test':testloader,'valid':validloader}
    return trainloader,testloader,validloader

def train_model(model,data):
    trainloader,testloader,validloader=def_dataloaders()
    if args.learning_rate is None:
        lr=0.002
    else:
        lr=args.learning_rate
     
    
    criterion=nn.NLLLoss()
    
    
    if args.gpu is None:
        device='cuda'
    else:
        device='cpu'
        
    
    optimizer=optim.Adam(model.classifier.parameters(),lr=float(lr))
    model.to(device)
    
    if args.training_epochs is None:
        epochs=2
    else:
        epochs=args.training_epochs
    
    steps=0
    check_every=30 
    running_loss=0
    
    #trainloader=loaders['train']
    for e in range(int(epochs)):
        for inputs,labels in trainloader:
            steps+=1
            inputs,labels=inputs.to(device) , labels.to(device)
        
            optimizer.zero_grad()
        
            output=model.forward(inputs)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
        
            if(steps % check_every ==0):
                valid_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                        for inputs,labels in validloader:
                            inputs,labels=inputs.to(device) , labels.to(device)
                        
                            log_op=model.forward(inputs)
                        
                            batch_loss=criterion(log_op,labels)
                            valid_loss+=batch_loss.item()
                        
                            op=torch.exp(log_op)
                            neglect,yhat=op.topk(1,dim=1)
                            equals= yhat==labels.view(*yhat.shape)
                        
                            accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/check_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                        
                running_loss=0
                model.train()
    running_loss=0
    print("End of training")
    return model

def save_model(model,args):
    checkpoint = {'state_dict': model.state_dict(),
                  'parameters':model.parameters(),
                  'class_to_index_dict':train_data.class_to_idx,
                  'arch':args.arch,
                  'hidden':args.hidden,
                  'epochs':args.epochs,
                  'device':args.gpu}
                  
    torch.save(checkpoint, 'checkpoint.pth')
    return 0


def build_model():
    
    arch_pretrained=args.arch
    hidden=args.hidden
    if arch_pretrained=='vgg':
        model=models.vgg16(pretrained=True)
        inputs=25088
    elif arch_pretrained=='densenet':
        model=models.densenet121(pretrained=True)
        inputs=1024
        
    
    for param in model.parameters():
        param.requires_grad=False
    #print(inputs,hidden)#,nn.Linear(inputs,hidden))
    newclassifier=nn.Sequential(nn.Linear(inputs,int(hidden)),
                             nn.ReLU(),
                             nn.Dropout(p=0.2),
                             nn.Linear(int(hidden),102),
                             nn.LogSoftmax(dim=1))
    
    model.classifier=newclassifier
    

    return model

def trainingmodel_building():
    data=data_loading() 
    #loaders=def_dataloaders()
    model=build_model()
    model=train_model(model,data)
    save_model(model)
    return 0

def parse():
    parser=argparse.ArgumentParser(description='Train a neural network, endless parameters')
    parser.add_argument('--dir',type = str, default = 'flowers/')
    parser.add_argument('--arch', help='use either densenet or vgg')
    parser.add_argument('--learning_rate',default=0.002)
    parser.add_argument('--hidden')
    parser.add_argument('--training_epochs')
    parser.add_argument('--gpu')
    args=parser.parse_args()
    return args

def main():
    print("Training...")
    global args
    args=parse()
    trainingmodel_building()
    print("model built!")
    return None  
main()

#example to input  python train.py --dir flowers/ --arch vgg --learning_rate 0.002 --hidden=1000 --training_epochs 2 --gpu cuda