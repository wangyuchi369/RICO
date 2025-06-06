#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
from tqdm import tqdm
import logging
import time
import json
from glob import glob
from os import path as osp
import pickle
import argparse



def call_gpt4o(orig_img_path, new_img_path, prompt):
    pass
    #! Please implement the function to call GPT-4o with the provided prompt and return the response.
    #! This function should handle the API call to GPT-4o, passing the original and reconstructed image paths along with the prompt.
    #! The function should return the revised caption generated by GPT-4o.




def get_text(orig_prompt, orig_img_path, new_img_path):

    prompt = f'''
We are working on a project that involves generating captions for images and using these captions to reconstruct the images. The process follows these steps:  

1. **Original Image (First Image):** A caption is generated based on this image.  
2. **Reconstructed Image (Second Image):** The generated caption is used as input for a text-to-image model to create this image.  

### **Your Task**  
Compare the **original** and **reconstructed** images, analyzing their differences to identify potential improvements for the original caption. Based on your observations, provide a **revised caption** that could enhance the reconstruction quality. 

### **Guidelines for Comparison**  
When analyzing the differences between the two images, consider the following aspects:  

- **Visual Details:** Color, shape, texture, and material of objects.  
- **Composition & Layout:** Object positioning, spatial relationships, and overall scene structure.  
- **Human Attributes (if applicable):** Pose, facial expression, skin tone, clothing, and hairstyle.  
- **Perspective & Style:** Type of image, camera angle, depth of field, lighting, and artistic style.  
- **Text in the Image:** Accuracy of any visible words, symbols, or signs.  
- **Image Quality:** Blurriness, artifacts, or inconsistencies in object rendering.  
- **World Knowledge:** Proper nouns or specific real-world references that should be preserved.  
- **Color Aesthetics:** Color palette, grading, and overall mood consistency.  

---

### **How to Improve the Caption**  
- **Add missing details** that were lost in reconstruction.  
- **Clarify ambiguous descriptions** to provide more precise information.  
- **Correct any inaccuracies** based on observed differences.  
- **Specify key attributes** (e.g., `"a red leather couch"` instead of `"a couch"`).  

Your revised caption should aim to **reduce discrepancies** between the original and reconstructed images while maintaining a natural and informative description. Your are encouraged to make revised caption less than 512 tokens.

Now I provide the original image, reconstructed image and the original caption: {orig_caption}.

Please give me the revised caption that you believe could enhance the reconstruction quality (namely, make the new reconstructed image more like the original one at pixel level) enclosed with <revised caption>. And provide your analysis enclosed with <analysis> after.'''

   






    for i in range(20):
        caption = call_gpt4o(orig_img_path, new_img_path, prompt)
        if caption != None and caption != '':
            break
    return caption


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="comparison of gpt4o")
    parser.add_argument('--orig_img_path', type=str, help='original_img_path')
    parser.add_argument('--new_img_path', type=str, help='new_img_path')
    parser.add_argument('--orig_caption', type=str, help='original_caption')
    args = parser.parse_args()
    orig_img_path = args.orig_img_path
    new_img_path = args.new_img_path
    orig_caption = args.orig_caption
    caption = get_text(orig_caption, orig_img_path, new_img_path)
    print(caption)