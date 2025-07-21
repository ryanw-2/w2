# rdp_utils.py
import math

def perpendicular_distance(point, line_start, line_end):
    """Calculate perpendicular distance from a point to a line."""
    if line_start == line_end:
        return math.dist(point, line_start)
    
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    num = abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den

def rdp(path, epsilon):
    """Simplify a polyline using the Ramer-Douglas-Peucker algorithm."""
    if len(path) < 2:
        return path

    dmax = 0.0
    index = 0
    for i in range(1, len(path) - 1):
        d = perpendicular_distance(path[i], path[0], path[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        first_half = rdp(path[:index+1], epsilon)
        second_half = rdp(path[index:], epsilon)
        return first_half[:-1] + second_half
    else:
        return [path[0], path[-1]]
