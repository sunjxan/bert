import copy
import torch.nn as nn

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class EncoderLayer(nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn

    def forward(self):
        pass

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self):
        pass

class DecoderLayer(nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn

    def forward(self):
        pass

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self):
        pass
