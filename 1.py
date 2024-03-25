以下是一个使用PyTorch实现Transformer作为Seq2Seq模型的基本代码模板。这个示例包括了Encoder、Decoder和完整的Transformer模型定义，以及一个简单的训练循环。请注意，这只是一个基础模板，实际应用中可能需要根据具体任务（如机器翻译、文本摘要等）进行相应的调整和扩展。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        encoder_layers = TransformerEncoderLayer(emb_dim, num_heads, hid_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.fc_out = nn.Linear(emb_dim, input_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        encoded = self.pos_encoder(embedded)
        encoded = self.transformer_encoder(encoded)
        output = self.fc_out(encoded)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, num_heads, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        decoder_layers = TransformerDecoderLayer(emb_dim, num_heads, hid_dim, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, n_layers)
        self.fc_out = nn.Linear(emb_dim, output_dim)

    def forward(self, trg, encoder_output):
        embedded = self.embedding(trg)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_decoder(embedded, encoder_output)
        output = self.fc_out(output)
        return output


class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, num_heads, hid_dim, n_layers, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, emb_dim, num_heads, hid_dim, n_layers, dropout)
        self.decoder = TransformerDecoder(output_dim, emb_dim, num_heads, hid_dim, n_layers, dropout)

    def forward(self, src, trg):
        encoder_output = self.encoder(src)
        output = self.decoder(trg, encoder_output)
        return output


# 示例训练循环
def train(model, src_tensor, trg_tensor, optimizer, criterion, device):
    model.train()
    src_tensor, trg_tensor = src_tensor.to(device), trg_tensor.to(device)
    optimizer.zero_grad()

    output = model(src_tensor, trg_tensor[:-1, :])
    output_dim = output.shape[-1]
    output = output.reshape(-1, output_dim)
    trg_tensor = trg_tensor[1:, :].reshape(-1)

    loss = criterion(output, trg_tensor)
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    # 示例参数
    input_dim = 10000  # 输入词汇表大小
    output_dim = 10000  # 输出词汇表大小
    emb_dim = 256
    num_heads = 8
    hid_dim = 512
    n_layers = 6
    dropout = 0.5
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerSeq2Seq(input_dim, output_dim, emb_dim, num_heads, hid_dim, n_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 假设我们有预处理好的源序列src_tensor和目标序列trg_tensor
    for epoch in range(epochs):
        for i, (src_tensor, trg_tensor) in enumerate(train_loader):
            loss = train(model, src_tensor, trg_tensor, optimizer, criterion, device)
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
```

在这个代码模板中：

1. `PositionalEncoding`类实现了位置编码，用于为输入序列添加位置信息。
2. `TransformerEncoder`和`TransformerDecoder`分别实现了Transformer的编码器和解码器部分。每个部分包含词嵌入层、位置编码层、多层Transformer编码器/解码器层以及输出全连接层。
3. `TransformerSeq2Seq`类将编码器和解码器组合成一个完整的Seq2Seq模型。
4. 示例训练循环展示了如何在单个批次上进行前向传播、计算损失、反向传播和优化权重。实际应用中，应使用数据加载器（如`torch.utils.data.DataLoader`）来加载和批量化处理数据。

请根据实际任务需求，如数据预处理、模型参数调整、评估指标等，对上述代码进行相应修改和扩展。