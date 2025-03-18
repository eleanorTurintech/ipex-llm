import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM as IPEXAutoModelForCausalLM

# Set environment variables to use Gaudi
os.environ["HABANA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all 8 Gaudi devices
os.environ["PT_HPU_ENABLE_LAZY_MODE"] = "1"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-67b-base")

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with IPEX-LLM optimization
print("Loading model with IPEX-LLM optimization...")
model = IPEXAutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-llm-67b-base",
    device_map="auto",  
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # ipex_config={
    #     "device": "hpu",
    #     "distributed": True,
    #     "distributed_type": "data_parallel"
    # }
)

# Print model device information
print(f"Model device configuration: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'Not available'}")

# Configure sequence length
max_length = 512

# Example inference
input_text = "Explain the concepts evolution and natural selection in simple terms that a high school student would understand."
print(f"Processing input: '{input_text}'")

# Properly prepare inputs with attention mask
inputs = tokenizer(
    input_text, 
    return_tensors="pt", 
    padding=True, 
    truncation=True,
    max_length=128
)

# Run generation
print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated response:")
print(response)