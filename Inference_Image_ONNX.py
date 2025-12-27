"""
Simple ONNX Inference for Images
Usage: python inference_image.py --model model.onnx --input image.jpg --output result.jpg
"""

import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
import os


def get_model_input_dtype(session):
    """
    Get the expected input dtype from ONNX model
    
    Returns:
        numpy dtype (np.float32, np.float16, or np.int8)
    """
    input_type = session.get_inputs()[0].type
    
    if 'float16' in input_type or 'Half' in input_type:
        return np.float16
    elif 'int8' in input_type:
        return np.int8
    else:
        return np.float32


def load_image(image_path, size=None, dtype=np.float32):
    """
    Load and preprocess image
    
    Args:
        image_path: path to input image
        size: resize to (width, height), None to keep original
        dtype: target numpy dtype
    
    Returns:
        image_tensor: numpy array [1, 3, H, W], normalized to [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    
    if size:
        img = img.resize(size, Image.BILINEAR)
    
    # Convert to numpy: (H, W, 3) -> (3, H, W)
    img_np = np.array(img).transpose(2, 0, 1).astype(dtype) / 255.0
    
    # Add batch dimension: (3, H, W) -> (1, 3, H, W)
    img_tensor = np.expand_dims(img_np, axis=0)
    
    return img_tensor


def save_image(tensor, output_path):
    """
    Save output tensor as image
    
    Args:
        tensor: numpy array [1, 3, H, W] or [3, H, W], range [0, 1]
        output_path: path to save image
    """
    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # (3, H, W) -> (H, W, 3)
    img_np = tensor.transpose(1, 2, 0)
    
    # Denormalize to [0, 255]
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    # Save
    img = Image.fromarray(img_np)
    img.save(output_path)
    print(f"✅ Saved result to: {output_path}")


def run_inference(model_path, input_path, output_path, size=None):
    """
    Run ONNX inference on a single image
    
    Args:
        model_path: path to ONNX model
        input_path: path to input image
        output_path: path to save output image
        size: resize to (width, height), None to keep original
    """
    print("="*60)
    print("ONNX Image Inference")
    print("="*60)
    
    # Check files
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Load ONNX model
    print(f"\n📦 Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Auto-detect input dtype
    input_dtype = get_model_input_dtype(session)
    dtype_name = {np.float32: 'FP32', np.float16: 'FP16', np.int8: 'INT8'}.get(input_dtype, 'Unknown')
    
    print(f"   Input:  {input_name}")
    print(f"   Output: {output_name}")
    print(f"   Expected dtype: {dtype_name}")
    
    # Load image
    print(f"\n🖼️  Loading image: {input_path}")
    image_tensor = load_image(input_path, size=size, dtype=input_dtype)
    print(f"   Shape: {image_tensor.shape}")
    print(f"   Dtype: {image_tensor.dtype}")
    
    # Run inference
    print(f"\n🚀 Running inference...")
    output = session.run([output_name], {input_name: image_tensor})[0]
    print(f"   Output shape: {output.shape}")
    
    # Save result
    print(f"\n💾 Saving result...")
    save_image(output, output_path)
    
    print("="*60)
    print("✅ Done!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Image Inference')
    parser.add_argument('--model', type=str, default='Pretrained_Weights/rt_focuser_wint8_afp16.onnx', help='Path to ONNX model')
    parser.add_argument('--input', type=str, default='Sample/Blurry.png', help='Path to input image')
    parser.add_argument('--output', type=str, default='Sample/Deblur_W8A16.png', help='Path to output image')
    parser.add_argument('--width', type=int, default=256, help='Resize width (optional)')
    parser.add_argument('--height', type=int, default=256, help='Resize height (optional)')
    
    args = parser.parse_args()
    
    # Resize if specified
    size = (args.width, args.height) if args.width and args.height else None
    
    run_inference(args.model, args.input, args.output, size=size)