#!/bin/bash
# setup.sh - Initializing the Recursive LLM Project

echo "🚀 Creating Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing PyTorch and SOTA Libraries..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🔧 Configuring DeepSpeed for Stage 3 Sharding..."
# Generates a basic deepspeed config for large context
cat <<EOF > ds_config.json
{
  "optimizer": { "type": "AdamW", "params": { "lr": 1e-4 } },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu" },
    "offload_param": { "device": "cpu" }
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
EOF

echo "✅ Setup Complete. Open Cursor and start with Phase 1."
