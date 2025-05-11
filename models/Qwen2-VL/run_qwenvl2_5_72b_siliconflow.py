import argparse
import base64
import datetime
import json
import os

from PIL import Image
import io
import requests


SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.getenv("SILICONFLOW_API_KEY")


PAYLOAD = {
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
        }
    ],
    "stream": False,
    "max_tokens": 512,
    "enable_thinking": False,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "stop": None,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": {"type": "text"},
}

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen2-VL 5.72B on SiliconFlow")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--prompt", type=str, help="Prompt for the model")
    parser.add_argument("--prompt_file", type=str, help="File containing the prompt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output")
    args = parser.parse_args()
    return args


def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='webp')
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode('utf-8')
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None


def run_qwenvl2_5_72b_siliconflow(image_dir, prompt, output_dir):
    output = dict()
    image_names = sorted(os.listdir(image_dir))

    for image in image_names:
        image_path = os.path.join(image_dir, image)
        image_base64 = convert_image_to_webp_base64(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{image_base64}",
                            "detail": "high",
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]

        payload = PAYLOAD.copy()
        payload["messages"] = messages

        print("=" * 50)
        print(f"Processing image: {image}")
        response = requests.post(SILICONFLOW_API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            result = response.json()
            output[image] = dict(prompt=prompt, response=result)
            print(result["choices"][0]["message"]["content"])
        else:
            print(f"Error: {response.status_code} - {response.text}")
        print("=" * 50)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"output_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    image_dir = args.image_dir
    prompt = args.prompt
    output_dir = args.output_dir
    if not prompt and not args.prompt_file:
        raise ValueError("Either --prompt or --prompt_file must be provided.")

    if args.prompt_file:
        print(f"Loading prompt from file: {args.prompt_file}")
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()

    run_qwenvl2_5_72b_siliconflow(image_dir, prompt, output_dir)
