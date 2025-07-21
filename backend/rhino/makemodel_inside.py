import compute_rhino3d.Util as util
import compute_rhino3d.Grasshopper as gh
import rhino3dm
import json
import base64

# Set up Rhino Compute server URL
util.url = "http://localhost:5000/"

# Input parameters
CSV_INPUT_PATH = "D:/W2 Workspace/paths.csv"
GH_FILE_PATH = "lines_to_model.gh"  # Path to your Grasshopper definition
OUTPUT_OBJ_PATH = "output_mesh.obj"  # Where to save the OBJ file

def run_grasshopper_script():
    try:
        # Read the CSV file content
        with open(CSV_INPUT_PATH, 'r') as file:
            csv_content = file.read()
        
        # Create input parameters for Grasshopper
        # Adjust parameter names to match your Grasshopper inputs
        inputs = {
            "CSV_Data": csv_content,  # Replace with your actual input parameter name
            # Add other inputs as needed:
            # "Parameter2": value2,
            # "Parameter3": value3,
        }
        
        # Evaluate the Grasshopper definition
        print("Running Grasshopper definition...")
        output = gh.EvaluateDefinition(GH_FILE_PATH, inputs)
        
        if output is None:
            print("Error: No output received from Grasshopper definition")
            return None
            
        print("Grasshopper evaluation completed successfully")
        
        # Extract mesh data from output
        # The output structure depends on your Grasshopper definition
        meshes = []
        
        for item in output:
            if hasattr(item, 'values') and item.values:
                for value in item.values:
                    if hasattr(value, 'data'):
                        # Decode base64 mesh data
                        mesh_data = base64.b64decode(value.data)
                        mesh = rhino3dm.CommonObject.Decode(mesh_data)
                        
                        if isinstance(mesh, rhino3dm.Mesh):
                            meshes.append(mesh)
                            print(f"Found mesh with {len(mesh.Vertices)} vertices")
        
        # Export meshes to OBJ format
        if meshes:
            export_to_obj(meshes, OUTPUT_OBJ_PATH)
            print(f"Successfully exported {len(meshes)} mesh(es) to {OUTPUT_OBJ_PATH}")
            return meshes
        else:
            print("No meshes found in Grasshopper output")
            return None
            
    except FileNotFoundError:
        print(f"Error: Could not find file {CSV_INPUT_PATH}")
        return None
    except Exception as e:
        print(f"Error running Grasshopper script: {str(e)}")
        return None

def export_to_obj(meshes, output_path):
    """Export meshes to OBJ format"""
    try:
        with open(output_path, 'w') as obj_file:
            vertex_offset = 0
            
            for i, mesh in enumerate(meshes):
                obj_file.write(f"# Mesh {i+1}\n")
                obj_file.write(f"g mesh_{i+1}\n")
                
                # Write vertices
                for vertex in mesh.Vertices:
                    obj_file.write(f"v {vertex.X} {vertex.Y} {vertex.Z}\n")
                
                # Write faces
                for face in mesh.Faces:
                    if face.IsQuad:
                        # Quad face (OBJ uses 1-based indexing)
                        obj_file.write(f"f {face.A + vertex_offset + 1} {face.B + vertex_offset + 1} {face.C + vertex_offset + 1} {face.D + vertex_offset + 1}\n")
                    else:
                        # Triangle face
                        obj_file.write(f"f {face.A + vertex_offset + 1} {face.B + vertex_offset + 1} {face.C + vertex_offset + 1}\n")
                
                vertex_offset += len(mesh.Vertices)
                
        print(f"OBJ file successfully written to {output_path}")
        
    except Exception as e:
        print(f"Error writing OBJ file: {str(e)}")

# Alternative method if your Grasshopper definition outputs data differently
def run_with_tree_structure():
    """Alternative approach for handling data tree outputs"""
    try:
        # For file path inputs, you might need to pass the path as a string
        inputs = {
            "FilePath": CSV_INPUT_PATH,  # If your GH script expects a file path
        }
        
        output = gh.EvaluateDefinition(GH_FILE_PATH, inputs)
        
        # Handle data tree structure
        for branch_index, branch in enumerate(output):
            print(f"Branch {branch_index}: {branch}")
            
            # Extract geometry from each branch
            if hasattr(branch, 'InnerTree'):
                for path, items in branch.InnerTree.items():
                    print(f"  Path {path}: {len(items)} items")
                    
                    for item in items:
                        if hasattr(item, 'data'):
                            # Process geometry data
                            geometry_data = base64.b64decode(item.data)
                            geometry = rhino3dm.CommonObject.Decode(geometry_data)
                            
                            if isinstance(geometry, rhino3dm.Mesh):
                                print(f"    Found mesh with {len(geometry.Vertices)} vertices")
                                # Process mesh...
                                
    except Exception as e:
        print(f"Error in tree structure method: {str(e)}")

if __name__ == "__main__":
    # Make sure Rhino Compute server is running
    print("Starting Rhino Compute Grasshopper evaluation...")
    print("Make sure Rhino Compute server is running on http://localhost:5000/")
    
    # Run the main function
    result = run_grasshopper_script()
    
    if result:
        print("Script completed successfully!")
    else:
        print("Script failed. Check the error messages above.")
        # Try alternative method
        print("\nTrying alternative tree structure method...")
        run_with_tree_structure()