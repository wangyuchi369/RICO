from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/pfs/Models/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="cuda"
# )


if __name__ == '__main__':
    # caption = qwen_single('/home/gaohuan03/wangyuchi/capae/datasets/capsbench/capsbench_img/1.jpg', 'Describe this image in detail. Your answer should be concise and informative.')
    # print(caption)
    import os
    import json
    prompt = 'Describe this image in detail. Your answer should be concise and informative.'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/gaohuan03/wangyuchi/capae/datasets/CompreCap/images')
    parser.add_argument('--output_dir', type=str, default='/home/gaohuan03/wangyuchi/capae/results/dpo_qwen/comprecap_orig_qwen.json')
    parser.add_argument('--model_path', type=str, default='/pfs/Models/Qwen2-VL-7B-Instruct')
    args = parser.parse_args()
    input_dir = args.input_dir
    # input_dir = '/home/gaohuan03/wangyuchi/capae/datasets/capsbench/capsbench_img'
    output = []
    from tqdm import tqdm
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    print('using model:', args.model_path)
    # default processer
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    

    # query = "Describe this image. Your answer should be concise and informative."
    def qwen_single(image_path, query):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
        return output_text[0]
    
    for img_file in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_file)
        caption = qwen_single(img_path, prompt)
        output.append({'image_file': img_path, 'caption': caption})
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    with open(args.output_dir, 'w') as f:
        json.dump(output, f, indent=4)
    