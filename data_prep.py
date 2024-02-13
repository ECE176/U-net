from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import cv2

class COCODataset(Dataset):
    def __init__(self, annotation_path, image_dir, transform=None):
        self.coco = COCO(annotation_path)
        self.image_dir = image_dir
        self.image_ids = self.coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        # Create a mask for each annotation
        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann) * ann['category_id'])
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask
