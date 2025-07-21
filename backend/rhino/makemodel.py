import compute_rhino3d.Util as util
import compute_rhino3d.Grasshopper as gh
import rhino3dm
import json

util.url = "http://localhost:5000/"

CSV_INPUT_PATH = "D:/W2 Workspace/paths.csv"

output = gh.EvaluateDefinition("lines_to_model.gh", CSV_INPUT_PATH)
print(output)
