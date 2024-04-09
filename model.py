import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-5):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.shape))
        self.bias = nn.Parameter(torch.empty(self.shape))
        self._reset_parameters()

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
    
    def _reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

class SublayerConnection(nn.Module):
    # norm和dropout都不改变shape
    def __init__(self, d_model, dropout=.1, norm_first=True):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        if self.norm_first:
            x = x + self.dropout(sublayer(self.norm(x)))
        else:
            x = self.norm(x + self.dropout(sublayer(x)))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = _get_clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=.1):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = _get_clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self._reset_parameters()

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        prob = self.generator(output)
        return prob

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # 计算query对key的相似度分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 将相似度分数转化为权重
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 使用权重对应value乘积累加，得到新的编码
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # (N,L,D) => (N,h,L,d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # (N,h,L,d_k) => (N,L,D)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.d_model)

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    # 不改变输入的shape
    def __init__(self, d_model, d_ff, dropout=.1, activation=F.relu):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(self.activation(self.lin1(x))))

def build_transformer(src_vocab, tgt_vocab, d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(d_model, nhead)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # (N,S,D)
    src_emb = Embeddings(d_model, src_vocab)
    # (N,T-1,D)
    tgt_emb = Embeddings(d_model, tgt_vocab)
    generator = Generator(d_model, tgt_vocab)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), num_layers),
        nn.Sequential(src_emb, c(position)),
        nn.Sequential(tgt_emb, c(position)),
        generator
    )
    return model