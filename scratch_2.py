import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

### requirements ##
#
# Python version 3.11.9
# Windows 10 22H2
#
# NVIDIA-SMI 576.80
# Nvidia Driver Version: 576.80
# GPU: NVIDIA GeForce GTX 1660 SUPER
# CUDA Version: 12.9
#
# cudnn_9.10.2_windows.exe
# cuda_12.9.0_576.02_windows.exe
# 576.80-desktop-win10-win11-64bit-international-dch-whql.exe
#
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install -U "triton-windows<3.3"
# pip install transformers
# pip install accelerate
#
###

# Hugging Face authentication
hf_access_token = "<insert token>"  # Replace with your actual token
login(token=hf_access_token)

# Set dtype based on GPU capability
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Load model only once
model_name = 'google/gemma-3-1b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch_dtype, device_map="cuda"
)

# Create the text-generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# System prompt and history
system_prompt = "You are a helpful assistant."


print("\nChatbot ready. Type 'exit' to quit.\n")

# Interactive loop
while True:
    history = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Exiting...")
        break

    # Append user message to history
    history.append({
        "role": "user",
        "content": [{"type": "text", "text": user_input}]
    })

    # Generate response
    output = pipe([history], max_new_tokens=300)[0][0]["generated_text"]

    # Extract assistant response from output
    assistant_message = next(
        (item for item in output if item["role"] == "assistant"), None
    )
    if assistant_message:
        response_text = assistant_message["content"]
        print(f"Assistant: {response_text}")
    else:
        print("Assistant: [No response generated]")
