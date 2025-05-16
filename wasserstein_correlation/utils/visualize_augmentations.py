import torch
from torch.utils.data import DataLoader
from wasserstein_correlation.utils.create_dataset import create_dataset
import kornia.augmentation as K  
import kornia.augmentation.container as KContainer
import matplotlib.pyplot as plt
from wasserstein_correlation.utils.aux_functions import extract_normalization, denormalize

root_dir = 'default_root_dir'

trainset, testset = create_dataset('STL10', root_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mean, std = extract_normalization(trainset)

testloader = DataLoader(
    testset, 
    batch_size=5,  
    shuffle=True,   
    num_workers=0
)

for _, batch in enumerate(testloader):
    data, _ = batch
    break

data = data.to(device)
augmentations = {
    'rotation': K.RandomRotation(degrees=180, p=1.0, same_on_batch=False),

    'translate': K.RandomTranslate(translate_x=(-0.3, 0.3), translate_y=(-0.3, 0.3), p=1.0, same_on_batch=False),

    'color': K.ColorJiggle(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=1.0, same_on_batch=False), 
    
    'perspective': K.RandomPerspective(distortion_scale=0.8, p=1.0, same_on_batch=False), 

    'affine_small': K.RandomAffine(
                degrees=(-30, 30),         
                translate=(0.2, 0.2),        
                scale=(0.8, 1.2),            
                shear=(-15, 15),             
                p=1.0,                       
                same_on_batch=False      
            ),

    'affine_large': K.RandomAffine(
        degrees=(-90, 90),          
        translate=(0.35, 0.35),        
        scale=(0.5, 1.2),            
        shear=(-25, 25),             
        p=1.0,
        same_on_batch=False
    ),

    'translate_rotate': K.RandomAffine(
        degrees=(-180, 180),          
        translate=(0.3, 0.3),                  
        p=1.0,
        same_on_batch=False
    ),

    'crop': K.RandomResizedCrop(size=(32, 32), scale=(0.5, 0.7), ratio=(0.75, 1.33), p=1.0, same_on_batch=False),

    'GaussianNoise': K.RandomGaussianNoise(mean=0.0, std=2.0, p=1.0, same_on_batch=False),

    'GaussianBlur': K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0, same_on_batch=False),
}

augmentation = augmentations['translate_rotate'] 


augmented_data = []
augmented_data.append(data)
for i in range(4): 
    augmented_data.append(augmentation(data))

augmented_data = torch.cat(augmented_data, dim=0)
augmented_data = denormalize(augmented_data, mean, std)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = augmented_data[i].cpu().permute(1, 2, 0).numpy()
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()