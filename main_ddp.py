
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from train import train

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, input_file, h_params):
    setup(rank, world_size)

    DEVICE = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model, all_loss, optimizer, encode, decode = train(input_file, h_params, DEVICE, rank, world_size)

    if rank == 0:  # Only let the main process print
        print(f"training_losses = {all_loss}")

    cleanup()

if __name__ == "__main__":
    input_file = 'dataset.txt'
    h_params = {
        'train_fraction': 0.1,
        'batch_size': 64,
        'block_size': 64,
        'n_episodes': 10,
        'eval_episode': 5,
        'learning_rate': 3e-4,
        'loss_estimate_episodes': 5,
        'n_embeddings': 384,
        'n_heads': 3,
        'n_layers': 3,
        'dropout': 0.2,
    }
    
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, input_file, h_params), nprocs=world_size, join=True)
