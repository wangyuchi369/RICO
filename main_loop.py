
from models_util.flux import generate_single_img_flux
# from models_util.cogvlm_single import cogvlm_single
from models_util.qwen_single import qwen_single
import json
from tqdm import tqdm
import os
import subprocess

INITIAL_PROMPT = '''Describe this image in detail. Your answer should be concise and informative.'''

ITER_STEPS = 2

def seperate_cap_any(caption):
    if caption.find('<analysis>') == -1:
        return caption, ''
    else:
        revised_caption = caption.split('<analysis>')[0].replace('<revised caption>', '').replace('</revised caption>', '').strip()
        analysis = caption.split('<analysis>')[1].replace('</analysis>', '').strip()
        return revised_caption, analysis

def get_init_caption(image_path, caption_json=None):
    if caption_json is not None:
        with open(caption_json, 'r') as f:
            init_caption = json.load(f)
            return init_caption[image_path]
    else:
        # return cogvlm_single(image_path, INITIAL_PROMPT)
        return qwen_single(image_path, INITIAL_PROMPT)

def t2i_model(caption, new_image_path):
    img = generate_single_img_flux(caption)
    img.save(new_image_path)
    if img is not None:
        return True
    return False

def update_caption(previous_caption, image_path, new_image_path):
    subprocess.run(['bash', 'models_util/environment_4o.sh'], check=True)
    result = subprocess.run(
        ['/usr/bin/python', 'models_util/gpt4o.py', '--orig_caption', previous_caption, '--orig_img_path', image_path, '--new_img_path', new_image_path],
        capture_output=True,
        text=True,
        check=True
    )        

    caption = result.stdout
    i = 0
    while caption is None or caption == '':
        print('Failed to generate caption.')
        result = subprocess.run(
            ['/usr/bin/python', 'models_util/gpt4o.py', '--orig_caption', previous_caption, '--orig_img_path', image_path, '--new_img_path', new_image_path],
            capture_output=True,
            text=True,
            check=True
        )

        caption = result.stdout
        i += 1
        if i > 10:
            caption = previous_caption
            break
    # print(caption)
    new_caption, analysis = seperate_cap_any(caption)
    return new_caption, analysis




def process_one(image_path, record, args):
    initial_caption = get_init_caption(image_path, args.caption_json)
    assert initial_caption is not None
    print(initial_caption)
    record['trajectory'] = []
    previous_caption = initial_caption
    img_name = image_path.split('/')[-1].split('.')[0]
    for i in tqdm(range(ITER_STEPS)):
        new_image_path = f"{args.output_video_dir}/{i+1}/{img_name}.jpg"
        os.makedirs(f"{args.output_video_dir}/{i+1}", exist_ok=True)
        success = t2i_model(previous_caption, new_image_path)
        if success:
            record['trajectory'].append({'index': i + 1, 'caption': previous_caption, 'image_path': new_image_path})
            updated_caption, analysis = update_caption(previous_caption, image_path, new_image_path)
            record['trajectory'][-1]['proposed_analysis'] = analysis
            previous_caption = updated_caption
            if i == ITER_STEPS - 1:
                record['trajectory'].append({'index': i + 2, 'caption': previous_caption})
        else:
            print(
                f"Failed to generate image for caption: {previous_caption}."
            )

    return record


def main_loop(args):
    # records = []
    sorted_images = sorted(os.listdir(args.image_folder))
    for image in tqdm(sorted_images[args.split::args.split_num]):
        if os.path.exists(f'{args.output_json_dir}/record_{image.split(".")[0]}.json'):
            continue
        image_path = os.path.join(args.image_folder, image)
        image_name = image.split('.')[0]
        print(image_name)
        record = {}
        record['image_path'] = image_path
        new_record = process_one(image_path, record, args)
        # records.append(new_record)
        os.makedirs(args.output_json_dir, exist_ok=True)
        with open(f'{args.output_json_dir}/record_{image_name}.json', 'w') as f:
            json.dump(record, f, indent=4)


if __name__ == '__main__':
    import argparse
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser = argparse.ArgumentParser(description="Comparison of GPT4O")
    parser.add_argument('--image_folder', type=str, default='datasets/capsbench')
    parser.add_argument('--caption_json', type=str, default=None)
    parser.add_argument('--output_video_dir', type=str, default='results/outputs')
    parser.add_argument('--output_json_dir', type=str, default='results/records')
    parser.add_argument('--split_num', type=int, default=1)
    parser.add_argument('--split', type=int, default=0)
    args = parser.parse_args()
    main_loop(args)
