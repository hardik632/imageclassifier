import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
def arg_parser():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--arch',type=str,help='Choose architecture')
    parser.add_argument('--save_dir',type=str,help='save directory for checkpoints')
    parser.add_argument('--learning_rate',type=float,help='Define gradient descent learning rate as float')
    parser.add_argument('--hidden_units',type=int,help='Hidden units')
    parser.add_argument('--epochs',type=int,help='Number of epochs')
    parser.add_argument('--gpu',action="store_true",help='GPU + Cuda')
    args = parser.parse_args()
    return args
def train_transformer(train_dir):
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
   training_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
   return training_datasets

def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    testing_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    return testing_datasets
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("CUDA was not found on device")
    return device
def pretrained_model(architecture="vgg16"):
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture

    for param in model.parameters():
        param.requires_grad = False 
    return model

def Classifier(model, hidden_units):
    if type(hidden_units) == type(None): 
        hidden_units = 4096 
    input_features = model.classifier[0].in_features
    classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(hidden_units, 1000),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(1000, 102),
                           nn.LogSoftmax(dim=1))
    return classifier
def validation(model, validation_dataloader, criterion):
    model.to('cuda')
    validity_loss = 0
    accuracy = 0
    for ii, (images, labels) in enumerate(validation_dataloader):
        
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        validity_loss += criterion(output, labels).item()
        
        probability = torch.exp(output)
        equality = (labels.data == probability.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validity_loss, accuracy
def training(model, training_dataloader, validation_dataloader, device, 
                  criterion, optimizer, epochs, print_every, steps):
    if type(epochs) == type(None): 
        epochs=20 
    model.to('cuda')
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(training_dataloader):
            
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validity_loss, accuracy = validation(model, validation_dataloader, criterion)
                
                    print("Epoch: {}/{} ".format(e+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Validation Loss: {:.3f} ".format(validity_loss/len(validation_dataloader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(validation_dataloader)))
                    running_loss = 0
                    model.train()

def validate_model(Model, Testloader, Device):
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy on test images is: %d%%' % (100 * correct / total))

def check_point(Model, Save_Dir, Train_data):

    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            model.class_to_idx = training_datasets.class_to_idx

            checkpoint = {'state_dict': model.state_dict(),
                          'classifier': model.classifier,
                          'class_to_idx': model.class_to_idx,
                          'optimizer_state': optimizer.state_dict,
                          'number_of_epochs': epochs}
            torch.save(checkpoint, 'checkpoint.pth')
def main():

    args = arg_parser()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    training_datasets = test_transformer(train_dir)
    validation_datasets = train_transformer(valid_dir)
    testing_datasets = train_transformer(test_dir)
    training_dataloader = data_loader(training_datasets)
    validation_dataloader = data_loader(validation_datasets, train=False)
    testing_dataloader = data_loader(testing_datasets, train=False)
    model = pretrained_model(architecture=args.arch)
    model.classifier = Classifier(model,hidden_units=args.hidden_units)
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print_every = 40
    steps = 0
    epochs=args.epochs
    trained_model = training(model, training_dataloader, validation_dataloader, 
                                  device, criterion, optimizer, epochs, 
                                  print_every, steps)
    validation_accuracy(trained_model, testing_dataloader, device)
    check_point(trained_model, args.save_dir, training_datasets)
if __name__ == '__main__': main()