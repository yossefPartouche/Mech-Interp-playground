import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        head_dim = q.shape[-1] #Formally D_k (but doesn't matter)
        T = q.shape[-2]
        attn_score = q @ k.transpose(-1, -2)
        scaled = attn_score / (head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T))
        masked = scaled.masked_fill(mask == 0, float('-inf'))
        ## CRUCIAL to choose the correct dimension to sum over
        ### mask.shape = (B, h, T, T) = (batch, head, query, token) so FOR EACH ROW we sum the columns
        p = masked.softmax(dim=-1)
        return p @ v


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # T is the number tokens/ in a single prompt
        # B is the number of prompts
        self.n_embd = n_embd # This is C for CONTEXT
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.attn = Attention()
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn .Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd)
        )

    def review_reshape(self, t):
        B, T, C = t.shape
        t_rev = t.view(B, T, self.n_head, C//self.n_head) # head_dim = C//self.n_head
        # Thinking: we want to split the tokens accross the attention heads 
        t_trans = t_rev.transpose(1, 2)
        # (B, self.n_head, T, head_dim)
        return t_trans

    def concat(self, t):
        #t.shape =  (B, self.n_head, T, head_dim)
        B, H, T, head_dim = t.shape
        t_trans = t.transpose(1, 2)
        t_cont = t_trans.contiguous()
        return t_cont.view(B, T, H*head_dim)
    
    def forward(self, x):
        print("x: ", x.shape)
        normed_in = self.ln1(x)
        qkv = self.qkv(normed_in)
        print("qkv:", qkv.shape)
        q_raw, k_raw, v_raw = qkv.split(self.n_embd, dim=-1)
        print("q_raw:", q_raw.shape)
        q = self.review_reshape(q_raw)
        print("q:", q.shape)
        k = self.review_reshape(k_raw)
        v = self.review_reshape(v_raw)

        attn_weights = self.attn(q, k, v)
        print("attn_weights: ", attn_weights)

        concatonated = self.concat(attn_weights)
        print("concatonated: ", concatonated)

        proj_out = self.out_proj(concatonated)
        print("proj_out: ", proj_out)
    
        residual_out1 = x + proj_out

        ln_out = self.ln2(residual_out1)
        ff = self.ff(ln_out)
        return residual_out1 + ff

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_blocks, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks_list = nn.ModuleList(Block(n_embd, n_head) for _ in range(num_blocks))
        self.ln = nn.LayerNorm(n_embd)
        self.out = nn.Linear(n_embd, vocab_size)


    def create_input(self, idx):
        _, T = idx.shape
        token_embd = self.wte(idx)
        pos_embd = self.wpe(torch.arange(T))
        merged = token_embd + pos_embd
        return merged


    def forward(self, idx):
        embeddings = self.create_input(idx)
        blocks_outputs = [embeddings]

        for b in self.blocks_list:
            blocks_outputs.append(b(blocks_outputs[-1]))

        depthed_out = blocks_outputs[-1]
        norm_out = self.ln(depthed_out)
        logits= self.out(norm_out)
        return logits


if __name__ == "__main__":
    model = Transformer(vocab_size=6, n_embd=4, block_size=4, num_blocks=2, n_head=2)
    idx = torch.tensor([[0, 1, 2, 3]])
    logits = model(idx)
    print(logits.shape)   # torch.Size([1, 4, 6]) = (B, T, V)