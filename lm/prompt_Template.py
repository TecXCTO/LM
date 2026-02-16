def format_sft_data(example):
    """Converts raw data into an Instruction-Following format."""
    return f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
