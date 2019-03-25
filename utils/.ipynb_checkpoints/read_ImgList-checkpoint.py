import os
import torch
import torch.utils.data as data
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class myImageFolder(data.Dataset):
    def __init__(self, root, ImageListFile, transform = None, target_transform=None, loader=default_loader):
        fh = open(ImageListFile)
        count = 0
        imgnlist=[]  # list of image names

        for line in fh.readlines():
            cls = line.split() 
            fn = cls.pop(0)   # image file name
            count += 1
            if os.path.isfile(os.path.join(root, '%s.jpg' % fn)):  # In our case, all image are jpg files
                imgnlist.append((fn, tuple([float(v) for v in cls])))
                
        print("%s lines detected, %s images loaded" % str(count) % str(myImageFolder.__len__()))
            
        self.root = root  # dir that stores images
        self.imgnlist = imgnlist
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, '%s.jpg' % fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label) 

    def __len__(self):
        return len(self.imgnlist)
    
    def getName(self):
        return self.classes
    

