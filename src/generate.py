import gc
import transformers
import torch
import diffusers
# 2 pipelines stable diffusion
from diffusers import TextToVideoSDPipeline, DPMSolverMultistepScheduler
from PIL import Image
import streamlit as st

@st.cache_resource
def make_pipeline(device:str, cpu_offload:bool, attention_slice:bool)->TextToVideoSDPipeline:
    # this pipeline for research only
    pipeline = TextToVideoSDPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float32 if device=="cpu" else torch.float16, variant="fp16", cache_dir = "./cache",
    )

    if cpu_offload:
        # this is should run on cpu mode only
        pipeline.enable_model_cpu_offload()
    if attention_slice is not None:
        pipeline.enable_vae_slicing()
    pipeline = pipeline.to(device)
    return pipeline


def generate(
        prompt,
        num_frames,
        num_steps,
        seed,
        height,
        width,
        device,
        cpu_offload,
        attention_slice
        ):
    pipeline = make_pipeline(device, cpu_offload, attention_slice)
    generator = torch.Generator(device=device).manual_seed(seed)

    video = pipeline(
        prompt,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        generator=generator,
        height=height,
        width=width,
    ).frames[0]
    # clear cache
    torch.cuda.empty_cache()
    gc.collect()
    return video

def upscale(video):
    pass