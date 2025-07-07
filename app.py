from exact_polylines import extract_geometry_from_sketch, save_paths_to_csv
from polyline_cleanup import clean_polylines

INPUT_FILEPATH = "./sSketch1.jpg"
OUTPUT_FILEPATH = "paths.csv"

wall_paths = extract_geometry_from_sketch(INPUT_FILEPATH, visualize_steps=True)
cleaned_paths = clean_polylines(wall_paths)

if wall_paths:
    save_paths_to_csv(wall_paths, OUTPUT_FILEPATH)