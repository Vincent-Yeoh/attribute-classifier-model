import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict,List,Tuple
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path 


from sklearn.preprocessing import MultiLabelBinarizer

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    return model_save_path
    
def load_json(filepath):
    with open(filepath, 'r') as file:
        content = json.load(file)
    return content
        
def crop_img(image, bbox):
    x = bbox[0]
    y= bbox[1]
    w = x + bbox[2]
    h= y + bbox[3]
    cropped_img = image.crop((x, y, w, h))
    return cropped_img



def plot_precision(x,Y,label_name):  
    plt.plot(x, Y, label=label_name)
    plt.xlabel('Epoch')
    plt.legend()
    
def plot_report(epoch_list):
    plt.figure(figsize=(15,15))
    
    x = range(len(epoch_list))
    
    label = np.array([[[value for key, value in report.items() if key != 'support'] for label, report in epoch.report.items()] for epoch in epoch_list])
    label_t = label.transpose()
    
    metrics_name = ['Precision', 'Recall', 'F1-Score']
    name= list(epoch_list[0].report.keys())
    split = len(label_t[0]) - 4
    
    plot_idx = 1
    for i in range(len(metrics_name)):
        title = metrics_name[i]
        plt.subplot(len(label_t), 2, plot_idx)
        for label in range(split):
            plot_precision(x=x, Y=label_t[i][label], label_name=name[label])
            plt.title(title + '(Labels)')

 
        plot_idx += 1
        
        plt.subplot(len(label_t), 2, plot_idx)
        for metric in range(split, len(label_t[0])):
                plot_precision(x=x,Y=label_t[i][metric],label_name=name[metric])
                plt.title(title + '(Aggregated Metrics)')
                
        plot_idx += 1
        
def calculate_pos_weight(dataset:torch.utils.data.Dataset):

    encodings = []
    for i in tqdm(range(len(dataset))):
        encodings.append(dataset[i][1])
    encodings = np.array(encodings)
    occurences = encodings.sum(axis=0)
    imb_per_ratio = occurences.max() / occurences
    mean_imb_per_ratio = imb_per_ratio.mean()
    max_imb_per_ratio = imb_per_ratio.max()
    CVIR = np.sqrt((np.power(imb_per_ratio - mean_imb_per_ratio, 2)/(len(occurences)-1)).sum())
    occurences, imb_per_ratio, mean_imb_per_ratio, max_imb_per_ratio, CVIR
    ##pos_weight - ratio of pos example over negative examples
    pos_counter = encodings.sum(axis = 0)
    pos_weight = (len(encodings) - pos_counter)/pos_counter
    return pos_weight
    
def inverse_transform(logits, classes:list):
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    return mlb.inverse_transform(logits.unsqueeze(dim=0))


#1. Create a function to take in a dataset
def display_random_images(dataset: torch.utils.data.Dataset,
                          classes = None,
                          n: int=10,
                          display_shape: bool = True,
                          seed: int = None):
    #2. Adjust display if n is too high
    if n > 5:
        display_shape = False
        print('Display_shape is set to False')
        if n > 10:
            n= 10
            print("For display purposes, n shouldn't be higher than 10, setting to 10")

        
    #3. Set the seed
    if seed:
            random.seed(seed)
        

    
    #4. Get random sampels indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)
        
        
        #5. Setup plot
    plt.figure(figsize=(16,8))

        #6 Loop through random indexes and plot them with mtplotlib
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        #7. adjust tensor dimensions for plotting
        targ_image_adjust = targ_image.permute(1,2,0)
        
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        if classes:
            
            title = f'{inverse_transform(targ_label, classes)}'
            if display_shape:
                title = title + f'\nshape: {targ_image_adjust.shape}'
            plt.title(title)
            plt.axis(False)
