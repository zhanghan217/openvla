"""
OpenVLA 推理测试脚本
适配 RTX 5090 / RTX 50-series (Blackwell sm_120)
使用 PyTorch 2.8.0+cu128 + SDPA 替代 flash-attn
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_path = "/home/wika/data/openvla/openvla-7b"

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print("Loading model...")
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    print("Model loaded!")

    image = Image.new("RGB", (224, 224), color="red")
    prompt = "In: What action should the robot take to pick up the red block?\nOut:"

    print("\nRunning inference...")
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(f"Action: {action}")
    print("\nDone!")


if __name__ == "__main__":
    main()
