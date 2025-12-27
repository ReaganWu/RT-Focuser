import torch
from PIL import Image
import torchvision.transforms as transforms
# Import RT-Focuser
from model.rt_focuser_model import RT_Focuser_Standard  # 替换为实际的文件名

# Load Model
model = RT_Focuser_Standard()
model.load_state_dict(torch.load(
    "Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth",
    map_location='cpu'
), strict=True)
model.eval()

# Load Img and Process
img = Image.open("Sample/Blurry.png").convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)

# save results
output_img = transforms.ToPILImage()(output.squeeze(0).cpu().clamp(0, 1))
output_img.save("Sample/Deblur_F32_Torch.png")
print("✅ Deblurred image saved to: /home/wuzy/project/Deblur/Deblurred_RT_Focuser.png")