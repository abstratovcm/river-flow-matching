import os
import torch
import imageio
import numpy as np
from lutils.configuration import Configuration
from lutils.logging import to_video
from model import Model

config = Configuration("configs/bair_64.yaml")
model = Model(config["model"])
model.load_from_ckpt("example/river_bair_64/model.pth")
model.cuda()
model.eval()

batch_size = 1
condition_frames = 1  # from training.condition_frames in the YAML
channels = 3        # assuming RGB images
img_size = config["data"]["input_size"]  # 64

# Create the initial noise image(s)
initial_images = torch.randn(batch_size, condition_frames, channels, img_size, img_size)

# Generate frames
generated_frames = model.generate_frames(
    initial_images.cuda(),  # of shape [b n c h w]
    num_frames=16,
    verbose=True)

# Convert generated frames to a video representation
generated_video = to_video(generated_frames)

# Define output folder and file paths
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
video_path = os.path.join(output_dir, "generated_video.mp4")
init_img_path = os.path.join(output_dir, "initial_image.png")

# Convert generated_video from (batch_size, num_frames, channels, h, w) to (num_frames, h, w, channels)
if hasattr(generated_video, "cpu"):
    generated_video = generated_video.cpu().numpy()
generated_video = generated_video[0].transpose(0, 2, 3, 1)

# Save the video using imageio
imageio.mimwrite(video_path, generated_video, fps=10)
print("Saved generated video to", video_path)

# --- Save the initial image ---
# Convert the initial image using a similar transformation as to_video.
# Here, initial_images has shape (batch_size, condition_frames, channels, h, w).
# We assume batch_size and condition_frames are 1, so we extract the single image.
initial_image_np = (((torch.clamp(initial_images, -1., 1.) + 1.) / 2.)
                    .detach().cpu().numpy() * 255).astype(np.uint8)
# Remove batch and condition dimensions: result shape (h, w, channels)
initial_image_np = initial_image_np[0, 0].transpose(1, 2, 0)

# Save the initial image as a PNG
imageio.imwrite(init_img_path, initial_image_np)
print("Saved initial image to", init_img_path)
