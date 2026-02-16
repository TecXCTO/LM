# LM
Language Model 

```
# How to Run Your New LLM System
Pre-process: Use the Hugging Face Tokenizers library to turn your text files into ID numbers.
Train: Run the train_step function from the previous step using a GPU.
Inference: Use torch.argmax(logits[:, -1, :], dim=-1) to predict the next word in a loop.
```
