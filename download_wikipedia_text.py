from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)
train_dataset = dataset['train']

with open('wikipedia_text.txt', 'w', encoding='utf-8') as f:
    for example in tqdm(train_dataset):
        f.write(example['text'] + '\n\n')
        