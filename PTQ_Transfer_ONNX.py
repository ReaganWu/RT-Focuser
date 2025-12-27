"""
Hybrid Quantization for RT-Focuser
- Weights: INT8 (fixed-point)
- Activations: FP16 (floating-point)
- Export to ONNX
- Validate with sliding window inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms


# ============================================================================
# Dataset, you could define your dataset here
# ============================================================================
class GoProFullSizeDataset(Dataset):
    """GoPro Dataset - FullSize w/o Cut"""
    def __init__(self, root_dir, mode='test'):
        self.root_dir = root_dir
        self.mode = mode
        self.blur_paths = []
        self.sharp_paths = []

        mode_dir = os.path.join(root_dir, mode)
        sequence_dirs = sorted(os.listdir(mode_dir))

        for seq in sequence_dirs:
            blur_dir = os.path.join(mode_dir, seq, 'blur_gamma')
            sharp_dir = os.path.join(mode_dir, seq, 'sharp')
            if not os.path.isdir(blur_dir): 
                continue

            blur_imgs = sorted(os.listdir(blur_dir))
            for img_name in blur_imgs:
                blur_path = os.path.join(blur_dir, img_name)
                sharp_path = os.path.join(sharp_dir, img_name)
                if os.path.exists(blur_path) and os.path.exists(sharp_path):
                    self.blur_paths.append(blur_path)
                    self.sharp_paths.append(sharp_path)

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur_path = self.blur_paths[idx]
        sharp_path = self.sharp_paths[idx]

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        blur_np = np.array(blur_img)
        sharp_np = np.array(sharp_img)
        
        blur_tensor = torch.from_numpy(blur_np).permute(2, 0, 1).float() / 255.0
        sharp_tensor = torch.from_numpy(sharp_np).permute(2, 0, 1).float() / 255.0

        return blur_tensor, sharp_tensor, blur_path


# ============================================================================
# Calibration Dataset (256x256 patches for calibration)
# ============================================================================
class CalibrationDataset(Dataset):
    """select the calibration set from test dataset"""
    def __init__(self, full_size_dataset, num_samples=100, patch_size=256):
        self.full_dataset = full_size_dataset
        self.num_samples = num_samples
        self.patch_size = patch_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # select img from dataset
        img_idx = idx % len(self.full_dataset)
        blur, sharp, _ = self.full_dataset[img_idx]
        
        C, H, W = blur.shape
        
        # random cut
        if H > self.patch_size and W > self.patch_size:
            top = torch.randint(0, H - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, W - self.patch_size + 1, (1,)).item()
            
            blur_patch = blur[:, top:top+self.patch_size, left:left+self.patch_size]
        else:
            # if too small, then resize it
            blur_patch = F.interpolate(
                blur.unsqueeze(0), 
                size=(self.patch_size, self.patch_size), 
                mode='bilinear'
            ).squeeze(0)
        
        return blur_patch


# ============================================================================
# Hybrid Quantization (Weights INT8, Activations FP16)
# ============================================================================
# ============================================================================
    
def apply_hybrid_quantization_v2(model, calibration_loader, device='cuda', 
                                  weight_dtype='int8', activation_dtype='fp16'):
    """
    Hybrid Quantization, enable multi combination quantization
    
    Args:
        model: PyTorch model
        calibration_loader: DataLoader for calibration
        device: 'cuda' or 'cpu'
        weight_dtype: 'int8' (quant) or 'fp32' (w/o quant)
        activation_dtype: 'int8', 'fp16', 'fp32'
    
    Returns:
        quantized_model: quantized rt-focuser
    """
    
    print("\n" + "="*80)
    print(f"Hybrid PTQ: Weights {weight_dtype.upper()} + Activations {activation_dtype.upper()}")
    print("="*80)
    
    model.eval()
    model.to(device)
    
    # ========== Step 1: model weights quantization ==========
    print(f"\n[1] Quantizing weights to {weight_dtype.upper()}...")
    
    if weight_dtype == 'int8':
        quantized_state_dict = {}
        original_state_dict = model.state_dict()
        
        for name, param in tqdm(original_state_dict.items(), desc="Quantizing weights"):
            if param.dtype in [torch.float32, torch.float16]:
                if 'weight' in name and param.dim() >= 2:
                    # Per-channel INT8 quantization
                    quantized_param = quantize_weight_int8(param)
                    quantized_state_dict[name] = quantized_param
                else:
                    # keep bias orignal data type
                    if activation_dtype == 'fp16':
                        quantized_state_dict[name] = param.half()
                    elif activation_dtype == 'fp32':
                        quantized_state_dict[name] = param.float()
                    else:  # int8
                        quantized_state_dict[name] = param.float()  # bias保持FP32
            else:
                quantized_state_dict[name] = param
        
        model.load_state_dict(quantized_state_dict)
    
    # ========== Step 2: activation quantization ==========
    print(f"\n[2] Converting activations to {activation_dtype.upper()}...")
    
    if activation_dtype == 'int8':
        model = wrap_model_for_int8_activation(model, calibration_loader, device)
    elif activation_dtype == 'fp16':
        model = model.half()
    elif activation_dtype == 'fp32':
        model = model.float()
    
    # ========== Step 3: Calibration ==========
    print(f"\n[3] Running calibration...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calibration_loader, desc="Calibration")):
            if i >= 50:
                break
            
            if activation_dtype == 'fp16':
                batch = batch.to(device).half()
            elif activation_dtype == 'fp32':
                batch = batch.to(device).float()
            else:  # int8
                batch = batch.to(device).float()
            
            _ = model(batch)
    
    print("Hybrid quantization completed!")
    print(f"   Weights: {weight_dtype.upper()}")
    print(f"   Activations: {activation_dtype.upper()}")
    
    return model


# ============================================================================
# INT8 act quant 
# ============================================================================

class QuantizedActivation(nn.Module):
    """INT8 act quant module"""
    def __init__(self):
        super().__init__()
        self.scale = None
        self.zero_point = None
        self.calibrated = False
        
    def calibrate(self, x):
        """get the statics info"""
        x_min = x.min().item()
        x_max = x.max().item()
        
        # calculate calibration value
        self.scale = (x_max - x_min) / 255.0
        self.zero_point = -x_min / self.scale
        self.zero_point = max(0, min(255, round(self.zero_point)))
        self.calibrated = True
        
    def forward(self, x):
        if not self.calibrated:
            # Calibration stage
            self.calibrate(x)
            return x
        
        # quant
        x_quantized = torch.round(x / self.scale + self.zero_point).clamp(0, 255)
        
        # dequant
        x_dequantized = (x_quantized - self.zero_point) * self.scale
        
        return x_dequantized


def wrap_model_for_int8_activation(model, calibration_loader, device):
    
    print("\n⚠️  Warning: INT8 activation quantization is simplified")
    print("   For production, use PyTorch's torch.quantization API")
    
    # simplyfy only change the input-output data type
    class QuantizedModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.input_quant = QuantizedActivation()
            self.output_quant = QuantizedActivation()
            
        def forward(self, x):
            x = self.input_quant(x)
            x = self.model(x)
            x = self.output_quant(x)
            return x
    
    wrapped_model = QuantizedModelWrapper(model).to(device)
    
    # Calibration
    print("Calibrating INT8 activation quantization...")
    wrapped_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calibration_loader, desc="INT8 Activation Calibration")):
            if i >= 20: 
                break
            batch = batch.to(device).float()
            _ = wrapped_model(batch)
    
    print("INT8 activation calibration done")
    
    return wrapped_model

def quantize_weight_int8(weight, num_bits=8):
    """
    Per-channel INT8 quantization for weights
    
    Args:
        weight: torch.Tensor, shape [out_ch, in_ch, ...] or [out_ch, in_ch]
        num_bits: quantization bits (default 8)
    
    Returns:
        quantized_weight: torch.Tensor (same shape, but quantized to INT8 then dequantized to FP16)
    """
    
    # Reshape to [out_ch, -1] for per-channel quantization
    original_shape = weight.shape
    weight_2d = weight.reshape(original_shape[0], -1)
    
    # Per-channel min/max
    w_min = weight_2d.min(dim=1, keepdim=True)[0]
    w_max = weight_2d.max(dim=1, keepdim=True)[0]
    
    # Quantization scale and zero-point
    scale = (w_max - w_min) / (2**num_bits - 1)
    scale = torch.clamp(scale, min=1e-8)  # avoid zero
    
    # Quantize
    w_quantized = torch.round((weight_2d - w_min) / scale).clamp(0, 2**num_bits - 1)
    
    # Dequantize (simulate INT8 perf，but store in FP16)
    w_dequantized = w_quantized * scale + w_min
    
    # Reshape back
    w_dequantized = w_dequantized.reshape(original_shape)
    
    return w_dequantized.half()  


def export_to_onnx_v2(model, output_path, activation_dtype='fp16', 
                       input_size=(1, 3, 256, 256), opset_version=13):
    """
    export onnx, accept different type of data type.
    """
    
    print("\n" + "="*80)
    print(f"Exporting to ONNX (activation: {activation_dtype.upper()})")
    print("="*80)
    
    model.eval()
    device = next(model.parameters()).device
    
    # generate dummy——input according to quantized model data type
    if activation_dtype == 'fp16':
        dummy_input = torch.randn(*input_size).to(device).half()
    elif activation_dtype == 'fp32':
        dummy_input = torch.randn(*input_size).to(device).float()
    else:  
        dummy_input = torch.randn(*input_size).to(device).float() # int8 version, float as input, and quantize as INT8
    
    print(f"\n[1] Input shape: {input_size}")
    print(f"[2] Input dtype: {dummy_input.dtype}")
    print(f"[3] Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"ONNX model exported to: {output_path}")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")
    
    return output_path

# ============================================================================
# ONNX Inference with Sliding Window
# ============================================================================
class ONNXSlidingWindowInference_v2:
    def __init__(self, onnx_path, window_size=256, overlap=32, device='cuda'):
        """
        ONNX version sliding windows inference
        """
        import onnxruntime as ort
        
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.device = device
        
        # Generate ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # detect the expection type
        input_type = self.session.get_inputs()[0].type
        if 'float16' in input_type or 'Half' in input_type:
            self.input_dtype = np.float16
            self.dtype_name = 'FP16'
        elif 'float' in input_type:
            self.input_dtype = np.float32
            self.dtype_name = 'FP32'
        elif 'int8' in input_type:
            self.input_dtype = np.int8
            self.dtype_name = 'INT8'
        else:
            self.input_dtype = np.float32 
            self.dtype_name = 'FP32 (default)'
        
        print(f"ONNX Runtime session created")
        print(f"   Providers: {self.session.get_providers()}")
        print(f"   Input dtype: {self.dtype_name}")
        print(f"   Input shape: {self.session.get_inputs()[0].shape}")
        
    def process_image(self, image):
        """
        process_single image

        Args:
            image: torch.Tensor [C, H, W]
        Returns:
            deblurred: torch.Tensor [C, H, W]
        """
        C, H, W = image.shape
        
        # get padding size
        pad_h = (self.stride - (H - self.window_size) % self.stride) % self.stride
        pad_w = (self.stride - (W - self.window_size) % self.stride) % self.stride
        
        # Padding
        img_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
        _, H_pad, W_pad = img_padded.shape
        
        # initialize output
        output = torch.zeros_like(img_padded)
        weights = torch.zeros_like(img_padded)
        
        # generate mask
        weight_mask = self._create_weight_mask()
        
        # sliding
        num_h = (H_pad - self.window_size) // self.stride + 1
        num_w = (W_pad - self.window_size) // self.stride + 1
        total_windows = num_h * num_w
        processed = 0
        
        for i in range(0, H_pad - self.window_size + 1, self.stride):
            for j in range(0, W_pad - self.window_size + 1, self.stride):
                # get windows
                window = img_padded[:, i:i+self.window_size, j:j+self.window_size]
                
                # ONNX Inference
                window_input = window.unsqueeze(0).numpy().astype(self.input_dtype)
                
                window_output = self.session.run(
                    [self.output_name], 
                    {self.input_name: window_input}
                )[0]
                
                window_output = torch.from_numpy(window_output).squeeze(0).clamp(0, 1)
                
                output[:, i:i+self.window_size, j:j+self.window_size] += window_output * weight_mask
                weights[:, i:i+self.window_size, j:j+self.window_size] += weight_mask
                
                processed += 1
                if processed % 10 == 0:
                    print(f"  Progress: {processed}/{total_windows} windows", end='\r')
        
        print() 
        
        
        output = output / (weights + 1e-8)
        
        output = output[:, :H, :W]
        
        return output
    
    def _create_weight_mask(self):
        mask = torch.ones(3, self.window_size, self.window_size)
        
        if self.overlap > 0:
            fade = torch.linspace(0, 1, self.overlap)
            
            mask[:, :self.overlap, :] *= fade.view(1, -1, 1)
            mask[:, -self.overlap:, :] *= fade.flip(0).view(1, -1, 1)
            mask[:, :, :self.overlap] *= fade.view(1, 1, -1)
            mask[:, :, -self.overlap:] *= fade.flip(0).view(1, 1, -1)
        
        return mask
    
def calculate_metrics(pred, target):
    """get the performance"""
    if isinstance(pred, torch.Tensor):
        pred = pred.permute(1, 2, 0).cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.permute(1, 2, 0).cpu().numpy()
    
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
    
    psnr_value = psnr(target, pred, data_range=1.0)
    ssim_value = ssim(target, pred, data_range=1.0, channel_axis=2)
    
    return psnr_value, ssim_value


def evaluate_onnx_model(onnx_path, dataset, num_samples=100, save_dir=None):
    """
    evaluate the onnx model' performance
    """
    
    print("\n" + "="*80)
    print(f"Evaluating ONNX Model on {num_samples} images")
    print("="*80)
    
    processor = ONNXSlidingWindowInference_v2(onnx_path, window_size=256, overlap=32)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    total_psnr = 0
    total_ssim = 0
    
    for idx in range(num_samples):
        blur, sharp, blur_path = dataset[idx]
        img_name = os.path.basename(blur_path)
        
        print(f"\n[{idx+1}/{num_samples}] Processing: {img_name}")
        print(f"  Image size: {blur.shape[1]}x{blur.shape[2]}")
        
        # sliding window inference
        deblurred = processor.process_image(blur)
        
        # calculate value
        psnr_value, ssim_value = calculate_metrics(deblurred, sharp)
        total_psnr += psnr_value
        total_ssim += ssim_value
        
        print(f"  PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
        
        # save output
        if save_dir:
            output_img = transforms.ToPILImage()(deblurred.cpu())
            save_path = os.path.join(save_dir, img_name)
            output_img.save(save_path)
    
    # get the avg value
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print("\n" + "="*80)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*80)
    
    return avg_psnr, avg_ssim


# ============================================================================
# Main Pipeline: Usage Example
# ============================================================================
if __name__ == "__main__":
    # ========== Config ==========
    # select the quant type：
    # 1. W8A16: weight_dtype='int8', activation_dtype='fp16'  (你当前的)
    # 2. W8A8:  weight_dtype='int8', activation_dtype='int8'  (新增)
    # 3. W8A32: weight_dtype='int8', activation_dtype='fp32'  (新增)
    
    WEIGHT_DTYPE = 'int8'      # 'int8' or 'fp32'
    ACTIVATION_DTYPE = 'fp16'  # 'int8', 'fp16', 'fp32'

    # ========== Config for Path ==========
    MODEL_PATH = "Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth"
    GOPRO_ROOT = "/home/wuzy/DATASET/debulur/gopro"
    ONNX_OUTPUT = f"saved/onnx/rt_focuser_w{WEIGHT_DTYPE}_a{ACTIVATION_DTYPE}.onnx"
    SAVE_DIR = f"results/w{WEIGHT_DTYPE}_a{ACTIVATION_DTYPE}"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CALIBRATION_SAMPLES = 1000
    NUM_TEST_SAMPLES = 100
    
    print("="*80)
    print("RT-Focuser Hybrid Quantization Pipeline")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"GoPro root: {GOPRO_ROOT}")
    
    # ========== Step 1: load model ==========
    print("\n[Step 1] Loading original model...")
    
    # use the RT-focuser as input model
    try:
        from model.rt_focuser_model import RT_Focuser_Standard
        model = RT_Focuser_Standard()
        
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure rt_focuser_model.py is available")
        exit(1)
    
    # ========== Step 2: prepare calibartion data ==========
    print("\n[Step 2] Preparing calibration dataset...")
    
    test_dataset = GoProFullSizeDataset(root_dir=GOPRO_ROOT, mode='test')
    calib_dataset = CalibrationDataset(test_dataset, num_samples=NUM_CALIBRATION_SAMPLES)
    calib_loader = DataLoader(calib_dataset, batch_size=8, shuffle=True, num_workers=4)
    
    print(f"Calibration dataset: {len(calib_dataset)} patches")
    
    # ========== Step 3: hybrid quantization (Reconfigurable) ==========
    quantized_model = apply_hybrid_quantization_v2(
        model, 
        calib_loader, 
        device=DEVICE,
        weight_dtype=WEIGHT_DTYPE,
        activation_dtype=ACTIVATION_DTYPE
    )
    
    # ========== Step 4: Export ONNX ==========
    onnx_path = export_to_onnx_v2(
        quantized_model, 
        ONNX_OUTPUT, 
        activation_dtype=ACTIVATION_DTYPE,
        input_size=(1, 3, 256, 256)
    )

    # ========== Step 5: ONNX Evaluation ==========
    print("\n[Step 5] Evaluating ONNX model...")
    
    avg_psnr, avg_ssim = evaluate_onnx_model(
        onnx_path=onnx_path,
        dataset=test_dataset,
        num_samples=NUM_TEST_SAMPLES,
        save_dir=SAVE_DIR
    )
    
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    print(f"ONNX model: {onnx_path}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*80)
