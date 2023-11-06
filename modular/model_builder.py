
import torch
import torchvision
from torch import nn
from torchinfo import summary

class TinyVGGModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape:int):
        super().__init__()
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2,
            )
        )
        
        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2,
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape),
        )
        
    def forward(self, x):
        return self.classifier(self.conv_stack_2(self.conv_stack_1(x)))
    
class EfficientNetB0V1(nn.Module):
    def __init__(self, output_shape: int, device: torch.device):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = torchvision.models.efficientnet_b0(weights=weights).to(device)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=1280, 
                            out_features=output_shape, 
                            bias=True)).to(device)
    def forward(self,x):
        return self.model(x)
        
        
    def summary(self):
        print(summary(model=self.model, 
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
        ) 


class EfficientNetB0V7(nn.Module):
    def __init__(self, output_shape: int, device: torch.device):
        super().__init__()
        weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = torchvision.models.efficientnet_b7(weights=weights).to(device)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=1280, 
                            out_features=output_shape, 
                            bias=True)).to(device)
    def forward(self,x):
        return self.model(x)
        
        
    def summary(self):
        print(summary(model=self.model, 
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
        ) 
        
