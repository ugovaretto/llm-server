from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

class ChatCompletionRequest(BaseModel):
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    messages: list
    temperature: float = 0.7
    max_tokens: int = 150

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

@app.post("/v1/chat/completions")
def chat_completion(request: ChatCompletionRequest):
    # Convert messages to prompt format
    prompt = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        temperature=request.temperature,
        max_new_tokens=request.max_tokens,
        do_sample=True if request.temperature > 0 else False
    )

    # Decode the response
    response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1717418918,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs["input_ids"][0]),
            "completion_tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
            "total_tokens": len(inputs["input_ids"][0]) + (len(outputs[0]) - len(inputs["input_ids"][0]))
        }
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
