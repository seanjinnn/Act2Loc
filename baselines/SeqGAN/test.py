# czc
import torch
from generator import Generator

gen = Generator(vocab_size=10, embedding_dim=16, hidden_dim=16, use_cuda=False)
samples = gen.sample(2, 10)
print(samples)
