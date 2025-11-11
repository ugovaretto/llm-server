from fastapi import FastAPI, HTTPException, Request
from transformers import TextStreamer
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
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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

    from fastapi.responses import StreamingResponse

    def generate_response():
        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_tokens = []

        for output in model.generate(
            **inputs,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            do_sample=True if request.temperature > 0 else False,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
            return_dict_in_generate=True,
            output_scores=False
        ):
            # Accumulate tokens
            generated_tokens.append(output)

            # Convert accumulated tokens to text
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Yield partial response
            yield f'data: {json.dumps({
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": None
                }],
                "usage": {
                    "prompt_tokens": len(inputs["input_ids"][0]),
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            })}\n\n'

        # Final completion message
        yield f'data: {json.dumps({
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(inputs["input_ids"][0]),
                "completion_tokens": len(generated_tokens),
                "total_tokens": len(inputs["input_ids"][0]) + len(generated_tokens)
            }
        })}\n\n'

    return StreamingResponse(generate_response(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
