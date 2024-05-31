# import glob
# import os

# root = '/home/work/YaiBawi/dataset/MOT17/images/train'

# file_paths = sorted(glob.glob('/home/work/YaiBawi/dataset/MOT17/images/train/*/img1/*'))
# print(len(file_paths))

# with open('my_mot17.txt', 'w') as f:
#     for file_path in file_paths:
#         f.write(file_path+'\n')

import torch
import os

pretrained = os.path.join(
    os.path.dirname(__file__),
    '../models/best.pt'
)

model = torch.load(pretrained, map_location=torch.device('cpu'))

print(model['model'].float().state_dict())

# load_state_dict