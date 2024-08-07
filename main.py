import torch
from train import train

# Hyperparameters
INPUT = 'dataset.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# HYPER_PARAMS = {
#     'train_fraction': 0.9,
#     'batch_size': 64,
#     'block_size': 256,
#     'n_episodes': 5000,
#     'eval_episode': 500,
#     'learning_rate': 3e-4,
#     'loss_estimate_episodes': 200,
#     'n_embeddings': 384,
#     'n_heads': 6,
#     'n_layers': 6,
#     'dropout': 0.2,
# }

HYPER_PARAMS = {
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

model, all_loss, optimizer, encode, decode = train(INPUT, HYPER_PARAMS, DEVICE)
print(f"training_losses = {all_loss}")
