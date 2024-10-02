import logging

import timm
import torch
import lightning as L
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from clearml import Task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

# Define data transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=validation_transform)



task = Task.init(project_name="ResNet", task_name="Experiment AdamW 512 transfecr") 
hyper_config = {
   'epochs': 10,
   'lr': 0.001,
   'batch_size': 256,
}

task.connect(hyper_config)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=hyper_config["batch_size"], shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=hyper_config["batch_size"], shuffle=False, num_workers=4)

class LightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.criterion = nn.CrossEntropyLoss()
        
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        # with torch.no_grad():
        #     total += labels.size(0)
        #     _, predicted = torch.max(outputs.data, 1)
        #     correct += (predicted == labels).sum().item()
        # accuracy = correct / total
        # logger.report_scalar(title='accuracy',series='train',iteration=epoch, value=accuracy)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("loss/val", loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=hyper_config["lr"])
        return optimizer

resnet_module = LightningModel()

trainer = L.Trainer(max_epochs=hyper_config["epochs"])
trainer.fit(model=resnet_module, train_dataloaders=train_loader, val_dataloaders=test_loader)
