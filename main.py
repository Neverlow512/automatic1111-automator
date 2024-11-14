#!/usr/bin/env python3

import os
import sys
import json
import time
import requests
import base64
import argparse
import logging
from tqdm import tqdm  # For progress bar

# For keyboard listener
try:
    import pynput
    from pynput import keyboard
except ImportError:
    print("The 'pynput' module is required for pause functionality. Installing it now...")
    os.system(f"{sys.executable} -m pip install pynput")
    import pynput
    from pynput import keyboard

paused = False  # Global variable to control pause state
keyboard_listener = None  # Global variable for keyboard listener

def on_press(key):
    global paused
    try:
        if key == keyboard.Key.space:
            paused = not paused
            state = "Paused" if paused else "Resumed"
            print(f"\n{state}...")
    except AttributeError:
        pass

def start_keyboard_listener():
    global keyboard_listener
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

def stop_keyboard_listener():
    global keyboard_listener
    if keyboard_listener is not None:
        keyboard_listener.stop()
        keyboard_listener = None

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
    lines = block.strip().split('\n')
    data = {}
    current_key = None
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            current_key = key.strip()
            data[current_key] = value.strip()
        else:
            if current_key:
                data[current_key] += ' ' + line.strip()
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

def get_available_models(sd_models_path):
    model_extensions = ('.ckpt', '.safetensors', '.pt')
    models = [f for f in os.listdir(sd_models_path) if os.path.isfile(os.path.join(sd_models_path, f)) and f.lower().endswith(model_extensions)]
    return models

def get_available_loras(sd_loras_path):
    lora_extensions = ('.ckpt', '.safetensors', '.pt')
    loras = [f for f in os.listdir(sd_loras_path) if os.path.isfile(os.path.join(sd_loras_path, f)) and f.lower().endswith(lora_extensions)]
    return loras

def get_available_samplers(api_endpoint):
    try:
        response = requests.get(f'{api_endpoint}/sdapi/v1/samplers')
        response.raise_for_status()
        samplers = response.json()
        return [sampler['name'] for sampler in samplers]
    except Exception as e:
        print(f"Error fetching samplers: {e}")
        return []

def get_available_schedulers(api_endpoint):
    try:
        response = requests.get(f'{api_endpoint}/sdapi/v1/schedulers')
        response.raise_for_status()
        schedulers = response.json()
        return [scheduler['name'] for scheduler in schedulers]
    except Exception as e:
        print(f"Error fetching schedulers: {e}")
        return []

def check_stable_diffusion_running(api_endpoint):
    # Check if the web UI is already running
    test_url = f'{api_endpoint}/sdapi/v1/sd-models'
    print(f"Checking if Stable Diffusion web UI is running at {test_url}...")
    try:
        response = requests.get(test_url)
        if response.status_code == 200:
            print("Stable Diffusion web UI is running.")
            return True
        else:
            print(f"Received unexpected status code: {response.status_code}")
            print(f"Response content: {response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False

def generate_json_files(prompts, prompt_type, story_name, default_seed):
    base_dir = os.path.join(story_name, 'Characters' if prompt_type == 'character' else 'Scenes')
    os.makedirs(base_dir, exist_ok=True)
    json_files = []
    for data in prompts:
        item_name = data.get('Name', 'Unnamed').replace(' ', '_')
        item_dir = os.path.join(base_dir, item_name)
        os.makedirs(item_dir, exist_ok=True)
        prompt_path = os.path.join(item_dir, 'prompt.json')
        # Remove 'Name' from data to avoid redundancy in JSON
        data_without_name = {k: v for k, v in data.items() if k != 'Name'}
        # Add placeholders for 'Number of Images', 'Number of Iterations', and 'Seed'
        data_without_name['Number of Images'] = 1
        data_without_name['Number of Iterations'] = 1
        data_without_name['Seed'] = default_seed
        with open(prompt_path, 'w') as f:
            json.dump(data_without_name, f, indent=4)
        json_files.append(prompt_path)
    return json_files

def generate_images(settings, prompt_type, story_name):
    api_url = settings.get('api_endpoint', 'http://localhost:7860') + '/sdapi/v1/txt2img'
    headers = {'Content-Type': 'application/json'}

    base_dir = os.path.join(story_name, 'Characters' if prompt_type == 'character' else 'Scenes')
    items = os.listdir(base_dir)
    for item_name in items:
        item_dir = os.path.join(base_dir, item_name)
        prompt_path = os.path.join(item_dir, 'prompt.json')
        with open(prompt_path, 'r') as f:
            data = json.load(f)

        num_images = int(data.get('Number of Images', 1))
        num_iterations = int(data.get('Number of Iterations', 1))
        seed = int(data.get('Seed', settings['seed']))

        print(f"\nGenerating images for {prompt_type}: {item_name}")
        print(f"Settings:")
        print(f"  Model: {settings['model']}")
        print(f"  LoRA: {settings['lora']}")
        print(f"  LoRA Weight: {settings['lora_weight']}")
        print(f"  Sampler: {settings['sampling_method']}")
        print(f"  Scheduler: {settings['scheduler']}")
        print(f"  Sampling Steps: {settings['sampling_steps']}")
        print(f"  Width: {settings['width']}")
        print(f"  Height: {settings['height']}")
        print(f"  CFG Scale: {settings['cfg_scale']}")
        print(f"  Seed: {seed}")
        print(f"  Number of Images: {num_images}")
        print(f"  Number of Iterations: {num_iterations}")

        for iteration in range(1, num_iterations + 1):
            iteration_dir = os.path.join(item_dir, f'Iteration_{iteration}')
            os.makedirs(iteration_dir, exist_ok=True)

            positive_prompt = data.get('Positive prompt', '')
            negative_prompt = data.get('Negative prompt', '')

            # Include LoRA settings at the end of the positive prompt
            if settings['lora']:
                lora_name = os.path.splitext(settings['lora'])[0]  # Remove file extension
                positive_prompt += f" <lora:{lora_name}:{settings['lora_weight']}>"

            # Check for pause
            while paused:
                time.sleep(0.5)

            payload = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "steps": settings["sampling_steps"],
                "cfg_scale": settings["cfg_scale"],
                "width": settings["width"],
                "height": settings["height"],
                "sampler_name": settings["sampling_method"],
                "seed": seed,
                "batch_size": 1,
                "n_iter": num_images,
                "scheduler": settings["scheduler"],
                "override_settings": {
                    "sd_model_checkpoint": settings["model"]
                }
            }

            # Log the payload
            logging.info(f"Generating images for {item_name}, Iteration {iteration}")
            logging.info(f"Payload: {json.dumps(payload, indent=4)}")

            print(f"\nIteration {iteration}: Generating {num_images} images...")
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                r = response.json()

                # Log the response
                logging.info(f"Response: {response.text}")

                for idx, img_data in enumerate(tqdm(r['images'], desc=f"Saving images for {item_name}")):
                    img_bytes = base64.b64decode(img_data)
                    img_path = os.path.join(iteration_dir, f'{item_name}_{iteration}_{idx + 1}.png')
                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_bytes)
                print(f"Iteration {iteration}: Completed generating images for {item_name}")
            except requests.exceptions.RequestException as e:
                print(f"Error generating images for {item_name} in iteration {iteration}: {e}")
                logging.error(f"Error generating images for {item_name} in iteration {iteration}: {e}")
                continue

            # Check for pause
            while paused:
                time.sleep(0.5)

def main():
    # Configure logging
    logging.basicConfig(filename='generation_log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    # Load Stable Diffusion settings
    sd_settings = load_sd_settings()

    # Command-line arguments for main.py
    parser = argparse.ArgumentParser(description='Generate images for characters and scenes.')
    args = parser.parse_args()

    # Check if Stable Diffusion web UI is running
    api_endpoint = sd_settings.get("api_endpoint", "http://localhost:7860")
    if not check_stable_diffusion_running(api_endpoint):
        print("Stable Diffusion web UI is not running.")
        print("Please start the web UI manually before running this script.")
        sys.exit(1)

    # Prompt user for story name
    story_name = input("Enter the name of your story: ").strip()

    # Get settings from user or settings file
    settings_file = os.path.join(story_name, 'settings.json')
    if os.path.exists(settings_file):
        # Load settings from file
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        print("Loaded settings from previous session.")
    else:
        # Get available models and LoRAs
        sd_folder = sd_settings['sd_folder']
        models_path = os.path.join(sd_folder, 'models/Stable-diffusion')
        loras_path = os.path.join(sd_folder, 'models/Lora')

        models = get_available_models(models_path)
        loras = get_available_loras(loras_path)

        # Select model
        print("\nAvailable Models:")
        for idx, model in enumerate(models):
            print(f"{idx + 1}: {model}")
        model_choice = int(input("Select a model by number: ")) - 1
        model = models[model_choice]

        # Select LoRA
        print("\nAvailable LoRAs:")
        for idx, lora in enumerate(loras):
            print(f"{idx + 1}: {lora}")
        lora_choice_input = input("Select a LoRA by number (press Enter to skip): ")
        if lora_choice_input.strip():
            lora_choice = int(lora_choice_input) - 1
            lora = loras[lora_choice]
            lora_weight = input("Enter the LoRA weight (default 1.0): ").strip() or "1.0"
            lora_weight = float(lora_weight)
        else:
            lora = ""
            lora_weight = 1.0

        # Get available schedulers
        schedulers = get_available_schedulers(api_endpoint)

        # Select scheduler
        print("\nAvailable Schedulers:")
        for idx, scheduler in enumerate(schedulers):
            print(f"{idx + 1}: {scheduler}")
        scheduler_choice = int(input("Select a scheduler by number: ")) - 1
        scheduler = schedulers[scheduler_choice]

        # Get available samplers
        samplers = get_available_samplers(api_endpoint)

        # Select sampler
        print("\nAvailable Samplers:")
        for idx, sampler in enumerate(samplers):
            print(f"{idx + 1}: {sampler}")
        sampler_choice = int(input("Select a sampler by number: ")) - 1
        sampling_method = samplers[sampler_choice]

        # Ask for other settings
        sampling_steps = int(input("Enter the number of sampling steps (default 50): ").strip() or 50)
        width = int(input("Enter the image width (default 512): ").strip() or 512)
        height = int(input("Enter the image height (default 768): ").strip() or 768)
        cfg_scale = float(input("Enter the CFG scale (default 7.5): ").strip() or 7.5)
        seed_input = input("Enter the seed (enter '-1' for random, default -1): ").strip() or "-1"
        seed = int(seed_input)

        # Save settings
        settings = {
            "model": model,
            "lora": lora,
            "lora_weight": lora_weight,
            "sampling_method": sampling_method,
            "scheduler": scheduler,
            "sampling_steps": sampling_steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "api_endpoint": api_endpoint
        }

        os.makedirs(story_name, exist_ok=True)
        settings_file = os.path.join(story_name, 'settings.json')
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        print("\nSettings saved for this story.")

    # Process prompts and generate JSON files
    print("\nProcessing character prompts...")
    character_prompts = create_prompts('character')
    character_json_files = generate_json_files(character_prompts, 'character', story_name, settings['seed'])

    print("\nProcessing scene prompts...")
    scene_prompts = create_prompts('scene')
    scene_json_files = generate_json_files(scene_prompts, 'scene', story_name, settings['seed'])

    print("\nJSON files for characters and scenes have been created.")
    print("You can review and edit them before proceeding.")
    print("Set 'Number of Images', 'Number of Iterations', and 'Seed' in each JSON file as desired.")

    input("\nPress Enter to start image generation...")

    # Start keyboard listener
    print("Press 'Space' at any time to pause/resume the script during image generation.")
    start_keyboard_listener()

    # Generate images
    print("\nStarting image generation for characters...")
    generate_images(settings, 'character', story_name)

    print("\nStarting image generation for scenes...")
    generate_images(settings, 'scene', story_name)

    # Stop keyboard listener after image generation
    stop_keyboard_listener()

    print("\nImage generation completed.")

if __name__ == '__main__':
    main()
