# The_SOTA_Training_Script.py



import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

def train_step(model, dataloader, device, epochs=1, lr=3e-4):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # BF16 is better for stability than FP16 on modern GPUs (A100, RTX 30/40/50 series)
    scaler = GradScaler() 
    
    # Learning Rate Scheduler (Cosine Decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Setup RoPE Cosine/Sine for the current sequence length
            # (Assuming model handles this internally as defined in previous step)
            
            optimizer.zero_grad(set_to_none=True) # Faster than zero_grad()

            with autocast(dtype=torch.bfloat16):
                # We need to calculate RoPE freqs for the specific sequence
                seq_len = inputs.size(1)
                cos, sin = model.rotary_emb(inputs, seq_len)
                
                logits = model(inputs, cos, sin)
                
                # Reshape for cross-entropy: (Batch * Seq, Vocab)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1)
                )

            # Backpropagation with Scaling
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Crucial for preventing "Exploding Gradients" in Transformers)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}")

# Usage Initialization
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TransformerBlock(...) # From previous code block
# loader = get_dataloader("my_data.txt")
# train_step(model, loader, device)
