from exact_polylines import get_skeleton, extract_geometry_from_sketch, save_paths_to_csv
from polyline_cleanup import clean_polylines
from image_generation import load_stable_diffusion_model, generate

INPUT_FILEPATH = "./data/sSketch1.jpg"
OUTPUT_FILEPATH = "paths.csv"
PROMPT = "Floor plan of modern beach resort in Miami"

canny_img = get_skeleton(INPUT_FILEPATH)
# wall_paths = extract_geometry_from_sketch(canny_img, visualize_steps=True)
# cleaned_paths = clean_polylines(wall_paths)

# if wall_paths:
#     save_paths_to_csv(wall_paths, OUTPUT_FILEPATH)


pipe = load_stable_diffusion_model()
generate(canny_img, pipe, PROMPT, visualize_steps=True)