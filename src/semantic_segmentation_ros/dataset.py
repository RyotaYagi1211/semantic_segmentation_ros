import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
import torch.nn.functional as F
import json
import cv2
from semantic_segmentation_ros.utils.data_utils import add_background


from semantic_segmentation_ros.utils.data_utils import get_rgb_img_tensor, get_labelme_mask_tensor
from semantic_segmentation_ros.utils.vis_utils import vis_img_mask
from semantic_segmentation_ros.utils.data_utils import get_rgb_img_tensor, get_labelme_edge_mask_multi



class SegDataset(Dataset):
    

    def __init__(self, path: str, labels: list, augmentations: dict, is_train: bool = True):
        self.labels = labels  # 例: ["edge"]
        self.img_path = os.path.join(path, "img")
        self.ann_path = os.path.join(path, "ann")
        self.imgs = list(sorted(os.listdir(self.img_path)))
        self.anns = list(sorted(os.listdir(self.ann_path)))

        self.augment = augmentations["enable"]
        self.trans = build_transform(augmentations)

        self.is_train = is_train

    def __len__(self) -> int:

        return len(self.imgs)

    def __getitem__(self, idx: int):##読み込みは出来てる
        # ####これはエッジ用
        # # #print(f"Loading index {idx}")
        # img_path = os.path.join(self.img_path, self.imgs[idx])
        # mask_path = os.path.join(self.ann_path, self.anns[idx])
        # #print(f"  img={img_path}, mask={mask_path}")

        # img = get_rgb_img_tensor(img_path)                        # (3,H,W) float
        # mask = get_labelme_edge_mask_multi(mask_path,
        #                            labels=["edge","edge2"],
        #                            line_thickness=2, dilate_iters=0)





        
        ####これは識別用
        img_path = os.path.join(self.img_path, self.imgs[idx])
        mask_path = os.path.join(self.ann_path, self.anns[idx])

        img = get_rgb_img_tensor(img_path)
        mask = get_labelme_mask_tensor(mask_path, self.labels)











        #print(f"  shapes: img={img.shape}, mask={mask.shape}")
        # 変換（v2 Transformsは tv_tensors.Image / Mask を推奨）
        
        
    #padあり
        #16の倍数でないときは必要
        # img = F.pad(img, (0, 0, 0, 8))       # 下方向に8ピクセルpadding
        # mask = F.pad(mask, (0, 0, 0, 8))     # 同じくマスクもpadding


        if self.is_train and self.augment:
            img, mask = self.trans(tv_tensors.Image(img), tv_tensors.Mask(mask))
        mask = add_background(mask)  # (C,H,W) LongTensor, C=クラス数+1
        #vis_img_mask(img, mask)  # デバッグ用
        return img, mask  



def build_transform(augmentations: dict) -> transforms.Compose:#ここで拡張
    """
    Builds a composed torchvision transformation based on augmentation settings.

    Inputs: augmentations (dict) - Augmentation settings as a dictionary.
    Outputs: transforms.Compose - A torchvision Compose object containing all the transformations.
    """
    transform_list = []

    if augmentations["affine"]["enable"]:
        transform_list.append(transforms.RandomAffine(degrees=augmentations["affine"]["rotate"], translate=augmentations["affine"]["translate"], scale=augmentations["affine"]["scale"], shear=augmentations["affine"]["shear"]))

    if augmentations["perspective"]["enable"]:
        transform_list.append(transforms.RandomPerspective(distortion_scale=augmentations["perspective"]["distortion_scale"], p=augmentations["perspective"]["p"]))

    if augmentations["elastic"]["enable"]:
        transform_list.append(transforms.ElasticTransform(alpha=augmentations["elastic"]["alpha"], sigma=augmentations["elastic"]["sigma"]))

    trans = transforms.Compose(transform_list)
    return trans

def add_background(mask: torch.Tensor) -> torch.Tensor:
    """
    Adds a background channel to the mask tensor.

    Inputs: mask (torch.Tensor) - The original mask tensor without the background channel.
    Outputs: torch.Tensor - Updated mask tensor with the background channel added.
    """
    background = torch.all(mask == 0, dim=0, keepdim=True)
    return torch.cat((mask, background), dim=0)
