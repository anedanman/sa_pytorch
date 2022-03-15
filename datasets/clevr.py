import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os


sizes = ['small', 'large']
materials = ['rubber', 'metal']
shapes = ['cube', 'sphere', 'cylinder']
colors = ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan']


def list2dict(inpt_list):
    return {inpt_list[i]: i for i in range(len(inpt_list))}


size2id = list2dict(sizes)
mat2id = list2dict(materials)
shape2id = list2dict(shapes)
color2id = list2dict(colors)


class CLEVR(Dataset):
    def __init__(self, images_path, scenes_path, max_objs=6, get_target=True):
        self.max_objs = max_objs
        self.get_target = get_target
        self.images_path = images_path
        
        with open(scenes_path, 'r') as f:
            self.scenes = json.load(f)['scenes']
        self.scenes = [x for x in self.scenes if len(x['objects']) <= max_objs]
        
        self.transform = transforms.Compose([
            transforms.CenterCrop((192, 192)),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        img = Image.open(os.path.join(self.images_path, scene['image_filename'])).convert('RGB')
        img = self.transform(img)
        target = []
        if self.get_target:
            for obj in scene['objects']:
                coords = ((torch.tensor(obj['3d_coords']) + 3.) / 6.).view(1, 3)
                size = F.one_hot(torch.LongTensor([size2id[obj['size']]]), 2)
                material = F.one_hot(torch.LongTensor([mat2id[obj['material']]]), 2)
                shape = F.one_hot(torch.LongTensor([shape2id[obj['shape']]]), 3)
                color = F.one_hot(torch.LongTensor([color2id[obj['color']]]), 8)
                obj_vec = torch.cat((coords, size, material, shape, color, torch.Tensor([[1.]])), dim=1)[0]
                target.append(obj_vec)
            while len(target) < self.max_objs:
                target.append(torch.zeros(19))
            target = torch.stack(target)
        return {
            'image': img*2 - 1,
            'target': target
        }
