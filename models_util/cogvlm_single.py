"""
This is a demo for using CogVLM2 in CLI using Single GPU.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
for multi-GPU, please use cli_demo_multi_gpus.py
"""
import torch
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/pfs/Models/cogvlm2-llama3-chat-19B"
# DEVICE = 'cuda:7' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Loading CogVLM')
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

# Argument parser
# parser = argparse.ArgumentParser(description="CogVLM2 CLI Demo")
# parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
# args = parser.parse_args()

# if 'int4' in MODEL_PATH:
#     args.quant = 4

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# Check GPU memory
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 48 * 1024 ** 3 and not args.quant:
    print("GPU memory is less than 48GB. Please use cli_demo_multi_gpus.py or pass `--quant 4` or `--quant 8`.")
    exit()


model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True
    ).eval().to(DEVICE)

print('Having loaded CogVLM')
def cogvlm_single(image_path, query):
    image = Image.open(image_path).convert('RGB')
    history = []
    if image is None:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            template_version='chat'
        )
    else:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version='chat'
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print("\nCogVLM2:", response)
    
    return response

if __name__ == '__main__':
    # cogvlm_single('datasets/capsbench/0.jpg', 'Describe this image in detail, and the description should be between 15 to 80 words.')
    import os
    from tqdm import tqdm
    import json
    input_dir = 'datasets/capsbench'
    prompt = '''
Describe the image with rich and detailed observations. Your response should be vivid and over 400 words. You may pay attention to the dimensions of overall, main subject, background, movement of main subject, style, camera movement, general, image type, text, color, position, relation, relative position, entity, entity size, entity shape, count, emotion, blur, image artifacts, proper noun (world knowledge), color palette, and color grading. 
'''
    res = []
    for img in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img)
        output_caption = cogvlm_single(img_path, prompt)
        res.append({'image_file': img_path, 'caption': output_caption})
        
    with open('results/cogvlm_updated.json', 'w') as f:
        json.dump(res, f, indent=4)
        
