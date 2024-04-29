import time
import torch
import torch.nn as nn

import config

from data import load_tokenizer, create_dataloader
from model import build_model

DEVICE = config.device

class TrainState:
    step = 0
    accum_step = 0
    samples = 0
    tokens = 0

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass

class DummyScheduler:
    def step(self):
        pass

def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * model_size ** (-.5) * min(step ** (-.5), step * warmup ** (-1.5))

def run_epoch(data_iter, iter_size, model, criterion, optimizer, scheduler,
    mode="train", accum_iter=1, print_iter=40, train_state=TrainState()):

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        mask_lm_out, next_sent_out = model.forward(batch.tokens, batch.segments, batch.mask)
        mask_lm_loss = criterion(mask_lm_out, batch.labels)
        next_sent_loss = criterion(next_sent_out, batch.is_next)
        loss = mask_lm_loss + next_sent_loss

        if mode == "train":
            loss.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if (i % print_iter == 0 or i + 1 == iter_size) and mode == "train":
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print("Epoch Step: %6d/%6d | Accumulation Step: %3d | Loss: %6.2f | Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                % (i+1, iter_size, n_accum, loss / batch.ntokens, tokens / elapsed, lr))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens, train_state

def train():

    tokenizer = load_tokenizer()
    model = build_model(config.vocab_size, config.d_model, \
        config.n_heads, config.n_layers, config.d_ff, config.dropout).to(DEVICE)

    criterion = nn.NLLLoss(ignore_index=0)

    train_dataloader, train_size = create_dataloader(config.train_file, \
        config.batch_size, config.max_padding, shuffle=True, drop_last=True)
    val_dataloader, val_size = create_dataloader(config.val_file, \
        config.batch_size, config.max_padding, shuffle=False, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr, betas=(.9, .98), eps=1e-9)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, config.d_model, factor=1, warmup=config.warmup)
    )
    train_state = TrainState()

    for epoch in range(config.num_epochs):
        model.train()
        print(f"Epoch {epoch+1} Training ====", flush=True)
        _, train_state = run_epoch(train_dataloader, train_size, model, criterion, optimizer, \
            lr_scheduler, mode="train", accum_iter=config.accum_iter, train_state=train_state)

        file_path = "%s%.2d.pt" % (config.file_prefix, epoch)
        torch.save(model.state_dict(), file_path)

        print(f"Epoch {epoch+1} Validation ====", flush=True)
        model.eval()
        sloss, _ = run_epoch(val_dataloader, val_size, model, criterion, \
            DummyOptimizer(), DummyScheduler(), mode="eval")
        print('Loss: %6.2f' % sloss)

    file_path = "%sfinal.pt" % config.file_prefix
    torch.save(model.state_dict(), file_path)

if __name__ == '__main__':
    train()
