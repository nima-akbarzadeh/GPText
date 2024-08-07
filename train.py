import torch
from tqdm import tqdm
from gpt import GPTLanguageModel, estimate_loss, get_batch


def prepare_model_and_dataset(input_file, h_params, device):

    # Read hyper-params
    train_fraction = h_params['train_fraction']
    n_embeddings = h_params['n_embeddings']
    n_heads = h_params['n_heads']
    dropout = h_params['dropout']
    n_layers = h_params['n_layers']
    block_size = h_params['block_size']
    learning_rate = h_params['learning_rate']

    # Read the dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get all unique characters and vocab size
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_fraction*len(data))
    train_data = data[:n]
    valid_data = data[n:]

    # Prepare the model
    model = GPTLanguageModel(vocab_size, n_embeddings, n_heads, dropout, n_layers, device, block_size)
    model = model.to(device)

    # Print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M model parameters')

    # Create an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return encode, decode, train_data, valid_data, model, optimizer


def training_loop(model, train_data, valid_data, optimizer, h_params, device):

    # Read hyper-params
    n_episodes = h_params['n_episodes']
    eval_episode = h_params['eval_episode']
    loss_estimate_episodes = h_params['loss_estimate_episodes']
    block_size = h_params['block_size']
    batch_size = h_params['batch_size']

    all_loss = []

    # Learning episodes
    for iter in tqdm(range(n_episodes), desc="Training Episodes"):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_episode == 0 or iter == n_episodes - 1:
            losses = estimate_loss(model, loss_estimate_episodes, train_data, valid_data, block_size, batch_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_data, 'train', valid_data, block_size, batch_size, device)

        # evaluate the loss
        _, loss = model(xb, yb)
        all_loss.append(loss.item())

        # Update the model
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return model, all_loss, optimizer


def train(input_file, h_params, device):

    encode, decode, train_data, valid_data, model, optimizer = prepare_model_and_dataset(input_file, h_params, device)
    model, all_loss, optimizer = training_loop(model, train_data, valid_data, optimizer, h_params, device)

    return model, all_loss, optimizer, encode, decode

def generate(model, decode, device):
    
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    # open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))