import typing
import torch
from affordance.data.utils.locate import Net as model
from lerf.data.utils.feature_dataloader import FeatureDataloader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np


image_list = [transforms.ToTensor()(Image.open('/scratch/ashwin/affordances/frame_00001.jpg'))]
print(image_list[0].shape)

class AffordanceDataLoader(FeatureDataloader):

    def __init__(
            self,
            cfg: dict,
            device: torch.device,
            image_list: torch.Tensor,
            cache_path: str = None
    ):
        assert "image_shape" in cfg
        self.num_classes = 36
        self.model_type = 'dino_vits8'
        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.prep = transforms.Compose([
            transforms.Resize([224, 224], antialias=None),
            transforms.Normalize(mean=self.mean, std= self.std)
        ])
        
        super().__init__(cfg, device, image_list, cache_path)
        

    def create(self, image_list):
        # image_list1=transforms.ToTensor()(Image.open('/scratch/ashwin/affordances/frame_00001.jpg'))
        extracter = model(aff_classes=36)
        extracter.eval()
        extracter.load_state_dict(torch.load('/scratch/ashwin/affordances/test_scripts/checkpoints/best_seen.pth.tar'))
        prep_image = self.prep(image_list)
        locate_embeds = []
        for image in tqdm(prep_image, desc='locate', total=len(prep_image), leave=False):
            descripters = extracter.embed_forward(image.unsqueeze(0))
            descripters = descripters[0, ...].permute(1, 2, 0)
            locate_embeds.append(descripters.cpu().detach())
        self.data = torch.stack(locate_embeds, dim=0)

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)

if __name__ == '__main__':
    cfg = {'image_shape' : [738, 994]}
    dl = AffordanceDataLoader(cfg, 'cuda', image_list, Path('/scratch/ashwin/affordances/test_scripts/locate.info'))
    ar = np.load('/scratch/ashwin/affordances/test_scripts/locate.info.npy')
    print(ar.shape)