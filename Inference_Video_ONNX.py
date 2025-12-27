"""
Simple ONNX Inference for Videos
Usage: python inference_video.py --model model.onnx --input video.mp4 --output result.mp4
"""

import argparse
import numpy as np
import onnxruntime as ort
import cv2
import os
from tqdm import tqdm


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


def preprocess_frame(frame, dtype=np.float32):
    """
    Preprocess video frame
    
    Args:
        frame: OpenCV BGR frame (H, W, 3)
        dtype: target numpy dtype
    
    Returns:
        tensor: numpy array [1, 3, H, W], normalized to [0, 1]
    """
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # (H, W, 3) -> (3, H, W)
    frame_np = frame_rgb.transpose(2, 0, 1).astype(dtype) / 255.0
    
    # Add batch dimension
    tensor = np.expand_dims(frame_np, axis=0)
    
    return tensor


def postprocess_frame(tensor):
    """
    Postprocess output tensor to frame
    
    Args:
        tensor: numpy array [1, 3, H, W] or [3, H, W], range [0, 1]
    
    Returns:
        frame: OpenCV BGR frame (H, W, 3)
    """
    # Remove batch dimension
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # (3, H, W) -> (H, W, 3)
    frame_np = tensor.transpose(1, 2, 0)
    
    # Denormalize to [0, 255]
    frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
    
    # RGB -> BGR
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    
    return frame_bgr


def run_video_inference(model_path, input_path, output_path, resize=None):
    """
    Run ONNX inference on a video
    
    Args:
        model_path: path to ONNX model
        input_path: path to input video
        output_path: path to save output video
        resize: resize to (width, height), None to keep original
    """
    print("="*60)
    print("ONNX Video Inference")
    print("="*60)
    
    # Check files
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
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
    
    # Open video
    print(f"\n🎬 Opening video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # Apply resize if specified
    if resize:
        width, height = resize
        print(f"   Resizing to: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n🚀 Processing video...")
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Resize if needed
        if resize:
            frame = cv2.resize(frame, resize)
        
        # Preprocess
        input_tensor = preprocess_frame(frame, dtype=input_dtype)
        
        # Inference
        output_tensor = session.run([output_name], {input_name: input_tensor})[0]
        
        # Postprocess
        output_frame = postprocess_frame(output_tensor)
        
        # Write frame
        out.write(output_frame)
        
        pbar.update(1)
    
    pbar.close()
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\n✅ Saved result to: {output_path}")
    print("="*60)
    print("✅ Done!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Video Inference')
    parser.add_argument('--model', type=str, default='Pretrained_Weights/rt_focuser_wint8_afp16.onnx', help='Path to ONNX model')
    parser.add_argument('--input', type=str, default='Sample/Test.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='Sample/output.mp4', help='Path to output video')
    parser.add_argument('--width', type=int, default=None, help='Resize width (optional)')
    parser.add_argument('--height', type=int, default=None, help='Resize height (optional)')
    args = parser.parse_args()
    
    # Resize if specified
    resize = (args.width, args.height) if args.width and args.height else None
    
    run_video_inference(args.model, args.input, args.output, resize=resize)