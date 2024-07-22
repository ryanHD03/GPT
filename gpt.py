import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 20
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 6
dropout = 0.5

# mapping from characters to integers
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', ' ']
vocab_size = len(vocab)
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

def encode(text: str) -> list[int]:
    '''Convert text to list of integer indices'''
    return [char_to_idx[ch] for ch in text]

def decode(indices: list[int]) -> str:
    '''Convert a list of integer indices back into text'''
    decoded_text =  ''.join([idx_to_char[idx] for idx in indices])
    decoded_text = reverse_sum(decoded_text)
    return decoded_text

def reverse_sum(problem: str) -> str:
    '''Reverse the sum part of addition problem'''
    parts = problem.split('=')
    addends = parts[0].strip()
    sum = parts[1][::-1].strip()
    decoded_text = f"{addends} = {sum}"
    return decoded_text

def generate_random_problem():
    '''Generate random addition problems'''
    a = str(torch.randint(0, 10000, (1,)).item())
    b = str(torch.randint(0, 10000, (1,)).item())
    c = str(int(a) + int(b))[::-1] # reverse output to simulate typical addition algorithm
    return f"{a} + {b} = {c}"

def get_batch():
    '''Generate batch of encoded problems and masked targets'''
    problems = [generate_random_problem() for _ in range(batch_size)]
    data = [encode(problem.ljust(block_size)) for problem in problems]
    x = torch.tensor(data, dtype=torch.long)
    y = x.clone()

    # mask input positions of a + b with -1 in the targets
    for i in range(batch_size):
        eq_str = decode(x[i].tolist())
        plus_pos = eq_str.index('+')
        equal_pos = eq_str.index('=')
        a_pos = plus_pos - 1
        b_pos = equal_pos - 1
        y[i, [a_pos, plus_pos, b_pos, equal_pos]] = -1

    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch()
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        weight = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        # decoder block
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        # weighted aggregation of values
        v = self.value(x)
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x) # applied on a per token level

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_table(torch.arange(T, device = device))
        x = token_embedding + pos_embedding
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.fc(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index = -1)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to fit within pos_embd_table
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#generated_indices = model.generate(context, max_new_tokens=50)[0].tolist()
#generated_text = decode(generated_indices)
#print(generated_text)

# test model
def test_addition_generation(model, num_problems = 10):
    for _ in range(num_problems):
        problem = generate_random_problem()
        encoded = encode(problem.ljust(block_size))  # pad to block_size
        context = torch.tensor([encoded], dtype=torch.long).to(device)
        
        # generate new tokens based on context
        generated_indices = model.generate(context, max_new_tokens = 50)[0].tolist()
        generated_text = decode(generated_indices)
        
        problem = reverse_sum(problem)
        print(f"Problem: {problem}")
        print(f"Generated Solution: {generated_text}")
        print("")

test_addition_generation(model, num_problems = 5)
