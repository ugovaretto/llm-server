from fastapi import FastAPI, HTTPException, Request
from typing import List
from transformers import TextStreamer
from pydantic import BaseModel
import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

class ChatCompletionRequest(BaseModel):
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    messages: list
    temperature: float = 0.7
    max_tokens: int = 150
    stream: bool = False

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest, req: Request = None):
    # Validate model
    if request.model != "meta-llama/Meta-Llama-3-8B-Instruct":
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Invalid model")
    
    # Check if streaming is requested (either via Accept header or stream parameter)
    accept_header = req.headers.get("accept", "") if req else ""
    is_streaming = "text/event-stream" in accept_header or request.stream
    
    # Convert messages to prompt format
    prompt = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if is_streaming:
        from fastapi.responses import StreamingResponse
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        def generate_response():
            # Use TextIteratorStreamer for real streaming
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Run generation in a separate thread
            generation_kwargs = {
                **inputs,
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens,
                "do_sample": True if request.temperature > 0 else False,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens as they arrive
            for text_chunk in streamer:
                # Skip empty chunks
                if not text_chunk or text_chunk.strip() == "":
                    continue
                    
                yield f'data: {json.dumps({
                    "id": "chatcmpl-abc123",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": text_chunk
                        },
                        "logprobs": None,
                        "finish_reason": None
                    }]
                })}\n\n'
            
            # Wait for generation to complete
            thread.join()
            
            # Send [DONE] message
            yield 'data: [DONE]\n\n'
        
        return StreamingResponse(generate_response(), media_type="text/event-stream", headers={"Content-Type": "text/event-stream"})
    else:
        # Non-streaming response
        outputs = model.generate(
            **inputs,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            do_sample=True if request.temperature > 0 else False
        )

        # Decode the response properly
        generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
        if isinstance(generated_tokens, torch.Tensor):
            # Convert tensor to list of integers
            token_list = [int(token) for token in generated_tokens.cpu().detach().tolist()]
        else:
            # If it's already a list or other iterable
            token_list = [int(token) for token in generated_tokens]
            
        response_text = tokenizer.decode(token_list, skip_special_tokens=True)

        return {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": int(time.time()),
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
                "completion_tokens": len(token_list),
                "total_tokens": len(inputs["input_ids"][0]) + len(token_list)
            }
        }

class ModelResponse(BaseModel):
    id: str
    object: str
    owned_by: str


class ListModelsResponse(BaseModel):
    object: str
    data: List[ModelResponse]


@app.get("/v1/models")
def list_models():
    return ListModelsResponse(
        object="list",
        data=[
            ModelResponse(
                id="meta-llama/Meta-Llama-3-8B-Instruct",
                object="model",
                owned_by="meta-llama"
            )
        ]
    )

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
