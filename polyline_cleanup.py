import numpy as np
from shapely.geometry import LineString, MultiLineString, GeometryCollection
from shapely.ops import linemerge, unary_union, snap

def is_line_almost_straight(polyline, tolerance=3):
    """
    Hausdorff Distance
    """
    if len(polyline) <= 2:
        return True
    
    line = LineString(polyline)
    straight = LineString([polyline[0], polyline[-1]])
    return line.hausdorff_distance(straight) < tolerance

def snap_endpoints(paths, snap_threshold=5):
    endpoints = []
    for path in paths:
        if len(path) > 2:
            endpoints.append(path[0])
            endpoints.append(path[-1])
    
    snapped_paths = []
    for path in paths:
        new_path = []
        for pt in path:
            pt_arr = np.array(pt)

            min_distance = float('inf')
            nearest = None
            for e in endpoints:
                distance = np.linalg.norm(pt_arr - np.array(e))
                if distance < min_distance:
                    min_distance = distance
                    nearest = e

            if nearest and min_distance < snap_threshold:
                new_path.append(tuple(nearest))
            else:
                new_path.append(tuple(pt))  
        snapped_paths.append(new_path)
    return snapped_paths

def clean_polylines(polylines, merge_tolerance=2, snap_tolerance=5, straighten_tolerance=3):
    cleaned = snap_endpoints(polylines, snap_tolerance)

    straightened = []
    for path in cleaned:
        if is_line_almost_straight(path, tolerance=straighten_tolerance):
            straightened.append([path[0], path[-1]])
        else:
            straightened.append(path)

    return straightened