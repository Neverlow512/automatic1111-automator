#!/usr/bin/env python3

import os
import sys
import json
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Setup the automated image generation tool.')
    args = parser.parse_args()

    # Ask for Stable Diffusion path
    sd_folder = input("Enter the path to your Stable Diffusion web UI folder: ").strip()

    # Create a settings directory to store configurations
    settings_dir = os.path.join(os.getcwd(), 'settings')
    os.makedirs(settings_dir, exist_ok=True)

    # Save the Stable Diffusion path in a settings file
    sd_settings = {
        "sd_folder": sd_folder,
        "api_endpoint": "http://localhost:7860"
    }
    sd_settings_path = os.path.join(settings_dir, 'sd_settings.json')
    with open(sd_settings_path, 'w') as f:
        json.dump(sd_settings, f, indent=4)

    # Copy template files into the current directory
    template_files = ['characters.txt', 'scenes.txt']
    for template_file in template_files:
        if not os.path.exists(template_file):
            with open(template_file, 'w') as f:
                f.write('')  # Create empty template files

    # Create the main.py script
    main_script_content = get_main_script()
    main_script_path = os.path.join(os.getcwd(), 'main.py')
    with open(main_script_path, 'w') as f:
        f.write(main_script_content)
    os.chmod(main_script_path, 0o755)  # Make the script executable

    print("\nSetup complete. You can now edit 'characters.txt' and 'scenes.txt' with your prompts.")
    print("When ready, run 'python main.py' to start the image generation process.")

def get_main_script():
    # This function returns the content of main.py as a string
    script = '''#!/usr/bin/env python3

import os
import sys
import json
import time
import requests
import base64
import argparse
from tqdm import tqdm  # For progress bar

def load_sd_settings():
    settings_dir = os.path.join(os.getcwd(), 'settings')
    sd_settings_path = os.path.join(settings_dir, 'sd_settings.json')
    if not os.path.exists(sd_settings_path):
        print("Error: sd_settings.json not found. Please run setup.py first.")
        sys.exit(1)
    with open(sd_settings_path, 'r') as f:
        return json.load(f)

def load_prompts(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return [block.strip() for block in content.split('---') if block.strip()]

def parse_prompt_block(block):
    lines = block.split('\\n')
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    return data

def create_prompts(prompt_type):
    if prompt_type == 'character':
        prompts_file = 'characters.txt'
    else:
        prompts_file = 'scenes.txt'

    prompt_blocks = load_prompts(prompts_file)
    prompts = []
    for block in prompt_blocks:
        data = parse_prompt_block(block)
        prompts.append(data)
    return prompts

def generate_images(settings, prompt_type, story_name):
    api_url = settings.get('api_endpoint', 'http://localhost:7860') + '/sdapi/v1/txt2img'
    headers = {'Content-Type': 'application/json'}

    prompts = create_prompts(prompt_type)
    if prompt_type == 'character':
        base_dir = os.path.join(story_name, 'Characters')
        name_key = 'Character Name'
    else:
        base_dir = os.path.join(story_name, 'Scenes')
        name_key = 'Scene Name'

    os.makedirs(base_dir, exist_ok=True)

    for data in prompts:
        item_name = data.get(name_key, 'Unnamed').replace(' ', '_')
        item_dir = os.path.join(base_dir, item_name)
        os.makedirs(item_dir, exist_ok=True)
        num_images = int(data.get('Number of Images', 1))
        num_iterations = int(data.get('Number of Iterations', 1))

        # Save prompt.json
        prompt_path = os.path.join(item_dir, 'prompt.json')
        with open(prompt_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Generating images for {prompt_type}: {item_name}")
        for iteration in range(1, num_iterations + 1):
            iteration_dir = os.path.join(item_dir, f'Iteration_{iteration}')
            os.makedirs(iteration_dir, exist_ok=True)

            if prompt_type == 'character':
                positive_prompt = generate_positive_prompt(data)
            else:
                positive_prompt = generate_scene_positive_prompt(data)

            negative_prompt = data.get('Negative Prompt', '')

            payload = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "steps": settings["sampling_steps"],
                "cfg_scale": settings["cfg_scale"],
                "width": settings["width"],
                "height": settings["height"],
                "sampler_name": settings["sampling_method"],
                "seed": settings["seed"],
                "batch_size": 1,
                "n_iter": num_images,
                "override_settings": {
                    "sd_model_checkpoint": settings["model"]
                }
            }

            if settings["lora"]:
                payload["override_settings"]["sd_lora"] = settings["lora"]

            print(f"Iteration {iteration}: Generating {num_images} images...")
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                r = response.json()
                for idx, img_data in enumerate(tqdm(r['images'], desc=f"Saving images for {item_name}")):
                    img_bytes = base64.b64decode(img_data)
                    img_path = os.path.join(iteration_dir, f'{item_name}_{iteration}_{idx + 1}.png')
                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_bytes)
                print(f"Iteration {iteration}: Completed generating images for {item_name}")
            except requests.exceptions.RequestException as e:
                print(f"Error generating images for {item_name} in iteration {iteration}: {e}")
                continue

            if settings["pause_between_generations"]:
                input("Press Enter to continue to the next iteration...")

def generate_positive_prompt(data):
    # Construct the positive prompt for characters
    gender = data.get('Gender', 'They')
    pronoun = 'He' if gender.lower() == 'male' else 'She' if gender.lower() == 'female' else 'They'
    prompt = f"{data.get('Character Name', '')}, age {data.get('Age', '')}, {data.get('Description', '')}, {data.get('Attributes', '')}. "
    prompt += f"{pronoun} wears {data.get('Clothing', '')}. "
    prompt += "Comic book-style illustration with a dark, gritty, realistic vibe."
    return prompt

def generate_scene_positive_prompt(data):
    # Construct the positive prompt for scenes
    prompt = f"{data.get('Description', '')} "
    prompt += "Comic book-style illustration with a dark, gritty, realistic vibe."
    return prompt

def main():
    # Load Stable Diffusion settings
    sd_settings = load_sd_settings()

    # Command-line arguments for main.py
    parser = argparse.ArgumentParser(description='Generate images for characters and scenes.')
    parser.add_argument('--model', type=str, help='Name of the model to use.')
    parser.add_argument('--lora', type=str, help='Name of the LoRA to use.')
    parser.add_argument('--sampling_method', type=str, help='Sampling method to use.')
    parser.add_argument('--sampling_steps', type=int, help='Number of sampling steps.')
    parser.add_argument('--width', type=int, help='Width of the generated images.')
    parser.add_argument('--height', type=int, help='Height of the generated images.')
    parser.add_argument('--cfg_scale', type=float, help='CFG scale.')
    parser.add_argument('--seed', type=int, help='Seed for image generation.')
    parser.add_argument('--pause', action='store_true', help='Pause between generations.')
    args = parser.parse_args()

    # Prompt user for story name
    story_name = input("Enter the name of your story: ").strip()

    # Load or initialize settings
    settings = {
        "model": args.model or input("Enter the model name (e.g., 'model.ckpt'): ").strip(),
        "lora": args.lora or input("Enter the LoRA name (press Enter to skip): ").strip(),
        "sampling_method": args.sampling_method or "Euler a",
        "sampling_steps": args.sampling_steps or 50,
        "width": args.width or 512,
        "height": args.height or 768,
        "cfg_scale": args.cfg_scale or 7.5,
        "seed": args.seed or -1,
        "pause_between_generations": args.pause or True,
        "api_endpoint": sd_settings.get("api_endpoint", "http://localhost:7860")
    }

    generate_images(settings, 'character', story_name)
    generate_images(settings, 'scene', story_name)

if __name__ == '__main__':
    main()
'''

    return script

if __name__ == '__main__':
    main()
