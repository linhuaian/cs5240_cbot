
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import os
from tqdm import tqdm
import numpy as np
# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


# Video
def fetch_video(ele: Dict, nframe_factor=2):
    if isinstance(ele['video'], str):
        def round_by_factor(number: int, factor: int) -> int:
            return round(number / factor) * factor

        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]

        video, _, info = io.read_video(
            video,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], nframe_factor)
        else:
            fps = ele.get("fps", 1.0)
            nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
        idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
        return video[idx]

video_path = "../kinetics-dataset/k400/train/"
video_names = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))][:20]
#video_path = "yoga.mp4"


questions = ["What is in this video?", "What is the person doing?", "What are the people doing in the background?", "Are people in the video focused or distracted?", "Can you tell the mood of the person in the video?", "Can you identify objects in the video?", "Is anyone holding any object, what are they holding?", "Are there any safety hazards in the place?", "Can you tell the profession of people in the video?"]
result = {}
for q in questions:
    result[q] = []
video_result = []
# sample uniformly 8 frames from the video

for video_name in tqdm(video_names):
    try:
        video_info = {"type": "video", "video": video_path + video_names[0], "fps": 1.0}
        video = fetch_video(video_info)
    except Exception as e:
        print(str(e))
        continue
    for question_asked in result.keys():
        conversation = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": question_asked},
        ],
    }
]
        print(conversation)
        # Preprocess the inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')

        # Inference: Generation of the output
        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        result[question_asked].append(str(output_text).replace(",", "").replace("\n", ""))
        print(output_text)
    video_result.append(video_name)
    print(f"{video_name} Done")
header = ["video_name"] + list(result.keys())
print(header)
np.savetxt(f"qwenvl_evaluation_output.csv", np.insert(np.column_stack([video_result] + list(result.values())),0,header, axis=0), delimiter = ", ", fmt = "%s")
