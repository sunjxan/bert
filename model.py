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

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.segment = nn.Embedding(3, d_model, padding_idx=0)

    def forward(self, x, segment_label):
        return self.emb(x) + self.segment(segment_label)

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

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # 计算query对key的注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # src attention，mask形状(1,S)，广播后(S,S)，左侧为True，右侧为False，用于重新编码时忽略pad，但pad也会重新编码
        # tgt attention，mask形状(T-1,T-1)，右侧或45度右上为False，用于重新编码时忽略pad和该词后面的词，但pad也会重新编码
        # tgt-src attention，mask形状(1,S)，广播后(T-1,S)，左侧为True，右侧为False，每个词根据src的value重新编码，可用于预测下一个词
        scores = scores.masked_fill(mask == 0, -1e9)
    # 将注意力分数转化为权重
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

class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class NextSentencePrediction(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class BERT(nn.Module):
    def __init__(self, encoder, embed, position, mask_lm, next_sent):
        super().__init__()
        self.encoder = encoder
        self.embed = embed
        self.position = position
        self.mask_lm = mask_lm
        self.next_sent = next_sent
        self._reset_parameters()

    def encode(self, tokens, segment, mask):
        tokens = self.embed(tokens, segment)
        tokens = self.position(tokens)
        return self.encoder(tokens, mask)
    
    def forward(self, tokens, segment, mask):
        memory = self.encode(tokens, segment, mask)
        return self.mask_lm(memory), self.next_sent(memory)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def build_model(vocab_size, d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(d_model, nhead)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = BERT(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Embeddings(d_model, vocab_size),
        PositionalEncoding(d_model, dropout),
        MaskedLanguageModel(d_model, vocab_size),
        NextSentencePrediction(d_model)
    )
    return model