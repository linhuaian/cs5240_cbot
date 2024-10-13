import av
import torch
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os 

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

model_name = "llava-hf/LLaVA-NeXT-Video-7B-hf"
model_name_out = "llava_next_vid_7b"
# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
processor = LlavaNextVideoProcessor.from_pretrained(model_name)

video_path = "../kinetics-dataset/k400/train/"
video_names = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))][:20]
#video_names = ["Bu9cc4d1bOs_000092_000102.mp4"]
questions = ["What is in this video?", "What is the person doing?", "What are the people doing in the background?", "Are people in the video focused or distracted?", "Can you tell the mood of the person in the video?", "Can you identify objects in the video?", "Is anyone holding any object, what are they holding?", "Are there any safety hazards in the place?", "Can you tell the profession of people in the video?"]
result = {}
for q in questions:
    result[q] = []
video_result = []
for video_name in tqdm(video_names):
    # Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
    # video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="karate.mp4", repo_type="dataset")
    try:
        container = av.open(video_path + video_name)
    except Exception as e:
        print(str(e))
        print(video_name)
        print(f"Unable to read video {video_name}")
        continue
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)
    for question_asked in result.keys():
        conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_asked},
                        {"type": "video"},
                        ],
                    },
                ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, videos=video, return_tensors="pt").to('cuda')
        out = model.generate(**inputs, max_new_tokens=100)
        response = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(response)
        result[question_asked].append(str(response[0]).replace(",", "").replace("\n", ""))
    video_result.append(video_name)
header = ["video_name"] + list(result.keys())
print(header)
np.savetxt(f"{model_name_out}_evaluation_output.csv", np.insert(np.column_stack([video_result] + list(result.values())),0,header, axis=0), delimiter = ", ", fmt = "%s")
