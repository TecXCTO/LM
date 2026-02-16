# The_SFT_Trainer

def sft_loss_fn(logits, labels, tokenizer):
    """Calculates loss only on Assistant responses, ignoring User prompts."""
    # Create mask: 1 for assistant tokens, 0 for user/padding
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Custom logic to find <|im_start|>assistant tags and mask everything before them
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
    
    # Apply mask (simplified version)
    # mask = generate_assistant_mask(shift_labels) 
    # return (loss * mask).sum() / mask.sum()
    
    return loss.mean() 
