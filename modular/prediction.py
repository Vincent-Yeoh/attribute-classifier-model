import numpy as np
import torch
import torchvision
from pathlib import Path
from modular import utils, model_builder
data_dir = Path('data/16kColorMulti/')
model_dir = Path('models')
device = 'cpu' if torch.cuda.is_available() else 'cuda'

Colors = {'white', 'black', 'green', 'blue', 'brown', 'gray', 'yellow', 'silver', 'red', 'dark', 'cream colored', 'pink', 'blond', 'orange', 'colorful', 'purple', 'maroon', 'golden', 'chrome', 'colorless',}
Shapes =  {'rounded', 'rectangular', 'arch shaped', 'lined', 'square', 'circular', 'cylindrical',}
Textures = {'patterned', 'striped', 'brick', 'grassy', 'fluffy', 'soft', 'pointy', 'tiled', 'floral', 'mesh', 'dotted', 'checkered', 'bushy',}
Materials = {'wooden', 'plastic', 'steel', 'glass', 'stone', 'concrete', 'asphalt', 'leather', 'paper', 'rubber', 'fabric', 'ceramic', 'cardboard', 'clay',}
Descriptions = {'modern', 'shiny', 'reflective', 'transparent', 'cluttered', 'bright', 'wet', 'stacked', 'folded', 'curved', 'framed', 'cushioned', 'digital', 'black lettered', 'barren', 'filled'}

combined_set = set()

for s in [Colors, Shapes, Textures, Materials, Descriptions]:
    combined_set.update(s)
classes = sorted(list(combined_set))
  
def get_report(label_output, conf:float, num_attribute:int):

   

    colors = get_category_dict(label_output=label_output, categories=Colors) 
    shapes = get_category_dict(label_output=label_output, categories=Shapes)
    materials = get_category_dict(label_output=label_output, categories=Materials)
    textures = get_category_dict(label_output=label_output, categories=Textures)
    descriptions = get_category_dict(label_output=label_output, categories=Descriptions)

    color_list = {key:str(value) for key, value in colors.items() if value > conf}
    shape_list ={key:str(value) for key, value in shapes.items() if value > conf} 
    
    materials_list ={key:str(value) for key, value in materials.items() if value > conf}
    textures_list ={key:str(value) for key, value in textures.items() if value > conf}
    descriptions_list =  {key:str(value) for key, value in descriptions.items() if value > conf}
    return {'Predicted Colors': color_list, 'Predicted Shapes': shape_list, 'Predicted Material': materials_list, 'Predicted Textures': textures_list, 'Predicted Descriptions': descriptions_list}


def get_category_dict(label_output, categories):
    return {key:value for key, value in label_output.items() if key in categories}

def predict(image_path, model_path, conf:float = 0.8, num_attribute:int = 3):

    model = model_builder.EfficientNetB0V1(output_shape=len(classes), device=device)
    model.load_state_dict(torch.load(f=model_path))
    image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255.

    transform = model.transforms
    image = transform(image)

    model.eval()
    with torch.inference_mode():
        image_pred = model(image.unsqueeze(dim=0).to(device))
        image_pred = torch.sigmoid(input=image_pred)

        image_pred = np.array(image_pred)
        sorted_output = np.argsort(-image_pred)[0]
        label_output = {classes[x]:image_pred[0][x] for x in sorted_output}
        return get_report(label_output=label_output, conf=conf, num_attribute=num_attribute)

