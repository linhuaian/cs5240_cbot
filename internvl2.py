import os
import torch
from huggingface_hub import login
from tqdm import tqdm
import os
login(token)

print("Logged IN")
from transformers import AutoTokenizer, AutoModel
tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)

print("Tokenizer loaded")

model = AutoModel.from_pretrained(
    'OpenGVLab/InternVideo2-Chat-8B',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).cuda()
print("Model Loaded")

from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

decord.bridge.set_bridge("torch")

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
print("Reading Video")
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
        video_tensor = load_video(video_path + video_name, num_segments=8, return_msg=False)
        video_tensor = video_tensor.to(model.device)
    except Exception as e:
        print(str(e))
        continue
    for question_asked in result.keys():
        chat_history= []
        response, chat_history = model.chat(tokenizer, '', question_asked, media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
        result[question_asked].append(str(response).replace(",", "").replace("\n", ""))
        print(response)
    
    video_result.append(video_name)
    print(f"{video_name} Done")
header = ["video_name"] + list(result.keys())
print(header)
np.savetxt(f"internvl2_evaluation_output.csv", np.insert(np.column_stack([video_result] + list(result.values())),0,header, axis=0), delimiter = ", ", fmt = "%s")


