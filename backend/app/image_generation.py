import cv2 as cv
from skimage.morphology import skeletonize
from PIL import Image
import numpy as np
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.models.controlnets.controlnet import ControlNetModel
import torch
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

def load_stable_diffusion_model():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    
    return pipe

def load_img(file_path, flipped=True):
    input_img = cv.imread(file_path)
    if input_img is None:
        print("Error: No input found at path.")
        return
    
    if flipped:
        input_img = cv.flip(input_img, 0)
    
    return input_img

def get_canny(input_img, visualize=False):
    if input_img is None:
        print("Error: No input found at path.")
        return
    low_threshold = 100
    high_threshold = 200

    img = cv.Canny(input_img, low_threshold, high_threshold)

    if visualize:
        cv.imshow("skeleton", img)
        cv.waitKey(0)
    cv.destroyAllWindows

    img = img[:,:, None]
    img = np.concatenate([img, img, img], axis = 2)
    canny_image = Image.fromarray(img)

    return canny_image


def get_skeleton(input_img, visualize=False):
    if input_img is None:
        print("Error: No input found")
        return
    
    gray = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    smoothed = cv.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    blurred = cv.GaussianBlur(smoothed, (5, 5), 0)
    binary_img = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10
    )
    
    binary_img[binary_img == 255] = 1
    skeleton_img = skeletonize(binary_img)
    skeleton_img = skeleton_img.astype(np.uint8) * 255
    
    if visualize:
        cv.imshow("skeleton", skeleton_img)
        cv.waitKey(0)
    cv.destroyAllWindows

    skeleton_img = Image.fromarray(skeleton_img)
    
    return skeleton_img
    
def generate(img, pipe, prompt, visualize_steps=False):
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    out_image = pipe(
        prompt, num_inference_steps=20, generator=generator, image=img
    ).images[0]

    out_image.save("output.png")




    
    