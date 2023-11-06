import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict,List,Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
import pathlib
from modular import utils
from pathlib import Path
import os
import numpy as np
from datasets import load_dataset


def create_dataloaders(image_dir: str, batch_size:int, train_transform:transforms.Compose = None, test_transform: transforms.Compose = None, target_transform=None):
    image_dataset_path = image_dir / 'image'
    image_dataset = list(pathlib.Path(image_dataset_path).glob('*.jpg'))
    image_dict = {path.stem: path for path in image_dataset}
    attribute_dict = utils.load_json(image_dir / 'attribute_dict.json')
    mlb = MultiLabelBinarizer()
    
    X = [{x['image_id']:x['bbox']} for x in attribute_dict]
    y = mlb.fit_transform([x['attribute'] for x in attribute_dict])
    
    classes = mlb.classes_.tolist()
    X = np.array(X)[..., np.newaxis]
    X_train, y_train, X_test, y_test = iterative_train_test_split(X,y,test_size=0.2)
    X_train, X_test = np.squeeze(X_train), np.squeeze(X_test)
    
    train_data = MultiLabelCustom(X=X_train, y=y_train, image_dict=image_dict, classes=classes, transform=train_transform, target_transform=target_transform)
    test_data = MultiLabelCustom(X=X_test, y=y_test, image_dict=image_dict, classes=classes, transform=test_transform, target_transform=target_transform)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True
                                    )
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True) 
    return train_dataloader, test_dataloader, classes


class MultiLabelCustom(Dataset):
    def __init__(self, X, y, image_dict, classes, transform=None, target_transform=None):
        self.image_ids = X
        self.output = y
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        self.image_dict = image_dict
   
    
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        for idx, bbox in self.image_ids[index].items():
            image_id = idx
            image_bbox = bbox
        image_path = self.image_dict[image_id]
        cropped_image = utils.crop_img(Image.open(image_path), image_bbox)
        return cropped_image
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, List]:
        image = self.load_image(index)
        labels = torch.tensor(self.output[index], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, labels
    

def one_hot_encode(labels, classes: list = None):
    mlb = MultiLabelBinarizer()
    if classes:
        mlb.fit(classes)
    else:
        mlb.fit(labels)
    return mlb.transform(labels)
