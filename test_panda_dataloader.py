import glob
import torch
from groot_dreams.dataloader import MultiVideoActionDataset

def test_panda_dataloader():
    # Gather all panda train subsets
    dataset_paths = glob.glob("/root/wise_dataset_0.3.2/no_noise_demo_1_round/config_*_train/lerobot_data")
    print(f"Found {len(dataset_paths)} dataset paths.")
    
    # Instantiate dataset
    # By default, num_frames=13, height=224, width=448 for our config
    dataset = MultiVideoActionDataset(
        dataset_path=",".join(dataset_paths),
        num_frames=13,
        height=224,
        width=448,
        data_split="train",
    )
    
    print(f"Dataset completely loaded with length {len(dataset)}")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    # Fetch a single batch
    batch = next(iter(dataloader))
    
    # Print shapes
    print("\nBatch Retrieved Successfully!")
    print(f"Video Shape: {batch['video'].shape}")
    print(f"State Shape: {batch['action'].shape if 'state' not in batch else batch['state'].shape}") 
    # Notice the combined state/action tensor behavior depending on ConcatTransform
    if 'state' in batch:
        print(f"State min: {batch['state'].min().item():.3f}, max: {batch['state'].max().item():.3f}")
    if 'action' in batch:
        print(f"Action min: {batch['action'].min().item():.3f}, max: {batch['action'].max().item():.3f}")
        
    print(f"Video min: {batch['video'].min().item():.3f}, max: {batch['video'].max().item():.3f}")

if __name__ == "__main__":
    test_panda_dataloader()
