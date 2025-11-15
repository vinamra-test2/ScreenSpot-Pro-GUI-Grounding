# copy from Aria-UI
import json
import os
import re
import tempfile
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

class AriaUIVLLMModel():
    def load_model(self, model_name_or_path="Aria-UI/Aria-UI-base"):
        self.sampling_params = SamplingParams(
            max_tokens=50,
            top_k=1,
            stop=["\n\n"],
            temperature=0
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name_or_path, 
                            trust_remote_code=True, 
                            use_fast=False
                        )
        self.model = LLM(
            model=model_name_or_path,
            tokenizer_mode="slow",
            dtype="bfloat16",
            trust_remote_code=True,
        )

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        full_prompt = f'Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to: {instruction}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": full_prompt,
                    }
                ],
            }
        ]
        
        message = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        outputs = self.model.generate(
            {
                "prompt_token_ids": message,
                "multi_modal_data": {
                    "image": [image],
                },
            },
            sampling_params=self.sampling_params,
        )
        for o in outputs:
            generated_tokens = o.outputs[0].token_ids
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        try:
            point = ast.literal_eval(response.replace("\n\n", "").replace("```", "").replace(" ", "").strip())
            x, y = point
            x = x / 1000
            y = y / 1000
            if 0 <= x <= 1 and 0 <= y <= 1:
                return {'point': [x, y], 'raw_response': response}
        except Exception as e:
            return {'point': None, 'raw_response': response}
        
        return {'point': None, 'raw_response': response}