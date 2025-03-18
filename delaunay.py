# Delaunay triangulation functions to be added
import math
from math import sqrt
import numpy as np
import random
import copy
from scipy.spatial import ConvexHull


def isInCircle(a, b, c, d):
    t = np.matrix([[1, 1, 1], [a[0], b[0], c[0]], [a[1], b[1], c[1]]])
    dt = np.linalg.det(t)
    if dt == 0:
        print("Aligned")
        return 
    m = np.matrix([[1, 1, 1, 1], [a[0], b[0], c[0], d[0]], [a[1], b[1], c[1], d[1]], 
                   [a[0]**2 + a[1]**2, b[0]**2 + b[1]**2, c[0]**2 + c[1]**2, d[0]**2 + d[1]**2]]])
    dm = np.linalg.det(m)
    return dt * dm <= 0

def Xmin(p):
    return min(p)

def Xmax(p):
    return max(p)

def Ymax(p):
    return max(p, key=lambda x: [x[1], x[0]])  # If two x[1] are the same, check x[0]

def Ymin(p):
    return min(p, key=lambda x: [x[1], x[0]])

def dist(p, q):
    return sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

def signedArea(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2

def isInTriangle(p, t):
    a1 = signedArea(t[0], t[1], p)
    a2 = signedArea(t[1], t[2], p)
    a3 = signedArea(t[2], t[0], p)

    if (a1 >= 0 and a2 >= 0 and a3 >= 0) or (a1 <= 0 and a2 <= 0 and a3 <= 0):
        return True
    else:
        return False

def sortedOrder(p, v):
    return sorted(p, key=lambda x: [x[0] * v[0] + x[1] * v[1], -x[0] * v[1] + x[1] * v[0]])

def maxV(p, v):
    return max(p, key=lambda x: [x[0] * v[0] + x[1] * v[1], -x[0] * v[1] + x[1] * v[0]])

def minV(p, v):
    return min(p, key=lambda x: [x[0] * v[0] + x[1] * v[1], -x[0] * v[1] + x[1] * v[0]])

def monotonePolygon(P, v):
    mi = minV(P, v)
    ma = maxV(P, v)
    left = []
    right = []

    for i in P:
        if signedArea(mi, ma, i) >= 0:
            left.append(i)
        else:
            right.append(i)
    return list(reversed(sortedOrder(left, v))) + sortedOrder(right, v)

def earTest(P, i):
    n = len(P)
    if signedArea(P[i - 1], P[i], P[(i + 1) % n]) < 0:
        return False
    
    for j in range(n):  # Could be optimized further, 3n unnecessary conditionals
        if j != (i - 1) % n and j != i and j != (i + 1) % n and isInTriangle(P[j], [P[(i - 1) % n], P[i], P[(i + 1) % n]]):
            return False
    return True

def findEar(P):
    for i in range(len(P)):
        if earTest(P, i):
            return i

def earClip(P):
    n = len(P)
    p = copy.deepcopy(P)
    T = []
    i = 0
    while n > 3:
        i = findEar(p)
        T.append([p[i - 1], p[i], p[(i + 1) % n]])  # Add triangle to the list
        p.pop(i)
        n -= 1       
    T.append(p)
    return T

def compareEdges(a, b):
    if a[0] in b and a[1] in b:
        return True
    return False

def triangleAngles(T):
    """
    Receives:
        T: Triangle
    Returns:
        Minimum angle of Triangle T.
    """
    sides = [dist(T[i], T[(i + 1) % 3]) for i in range(3)]
    angles = [math.degrees(math.acos((sides[i] ** 2 + sides[(i + 1) % 3] ** 2 - sides[(i + 2) % 3] ** 2) / 
                                      (2 * sides[i] * sides[(i + 1) % 3]))) for i in range(3)]
    return angles

def edgeInTriangle(a, T):
    if a[0] in T and a[1] in T:
        return True
    return False

def getLegalEdge(ady, T):
    """
    Returns the legal edge of the two possible triangles, given the original triangle.
    """ 
    m_angle1 = min(triangleAngles([ady[0], ady[1], ady[3]]) + triangleAngles([ady[1], ady[2], ady[3]]))
    m_angle2 = min(triangleAngles([ady[0], ady[1], ady[2]]) + triangleAngles([ady[0], ady[2], ady[3]]))
    if signedArea(ady[0], ady[2], ady[1]) * signedArea(ady[0], ady[2], ady[3]) > 0:
        return sorted([ady[1], ady[3]]), [[ady[0], ady[1], ady[3]], [ady[1], ady[2], ady[3]]]
    if signedArea(ady[1], ady[3], ady[0]) * signedArea(ady[1], ady[3], ady[2]) > 0:
        return sorted([ady[0], ady[2]]), [[ady[0], ady[1], ady[2]], [ady[0], ady[2], ady[3]]]
    if m_angle1 > m_angle2: 
        return sorted([ady[1], ady[3]]), [[ady[0], ady[1], ady[3]], [ady[1], ady[2], ady[3]]]
    else:
        return sorted([ady[0], ady[2]]), [[ady[0], ady[1], ady[2]], [ady[0], ady[2], ady[3]]]

def getAdjacent(T1, T2):
    """
    Returns the polygon formed by the four vertices.
    """
    try:
        t0 = set([tuple(i) for i in T1])
        t1 = set([tuple(i) for i in T2])
        intersection = t0 & t1
        union = t0 | t1
        disjoint = union.difference(intersection)
        intersection = list(intersection)
        disjoint = list(disjoint)
        adj = [intersection[0], disjoint[0], intersection[1], disjoint[1]]
        return [list(i) for i in adj]
    except:
        print(T1, T2)
        return []

def getTriangleEdges(t):
    """
    Returns a list of edges where the points in edges are ordered arbitrarily.
    This way, we can later perform np.unique without added complexity.
    """
    return [sorted((t[0], t[1])), sorted([t[0], t[2]]), sorted([t[1], t[2]])]

def plotEdge(edge, color):
    x = [edge[0][0], edge[1][0]]
    y = [edge[0][1], edge[1][1]]
    return line([(x[0], y[0]), (x[1], y[1])], color=color, thickness=2)

def superTriangle(points):
    x_min = Xmin(points)[0]
    x_max = Xmax(points)[0]
    y_min = Ymin(points)[1]
    y_max = Ymax(points)[1]
    
    dx = x_max - x_min
    dy = y_max - y_min
    factor = 0.2
    vertices = [[(x_min - dx * (1 - factor)), y_min - dy * factor],
                [(x_max + dx * (1 - factor)), y_min - dy * factor],
                [(x_max + x_min) / 2, y_max + dy]]
    
    return [(vertices[0], vertices[1], vertices[2])]

def identifyBadTriangles(triangulation, point):
    bad_triangles = []
    for triangle in triangulation:
        if isInCircle(*triangle, point):
            bad_triangles.append(triangle)
    return bad_triangles

def identifyPolygon(bad_triangles):
    polygon = []
    for triangle in bad_triangles:
        for i in range(len(triangle)):
            shared = False
            edge = [triangle[i], triangle[(i + 1) % 3]]
            for t2 in bad_triangles:
                if t2 != triangle:
                    for j in range(len(t2)):
                        edge2 = [t2[j], t2[(j + 1) % 3]]
                        if edge == edge2 or edge == edge2[::-1]:
                            shared = True
                            break
            if not shared:
                polygon.append(edge)
    return polygon

def incrementalAlgorithm(points):
    first_triangle = superTriangle(points)
    triangulation = first_triangle.copy()
    for point in points:
        s = point(points, color='black', size=30) + sum(line([t[0], t[1], t[2], t[0]]) for t in triangulation) + point(point, color='red')
        show(s, figsize=2.5, axes=False)
        bad_triangles = identifyBadTriangles(triangulation, point)
        s = point(points, color='black', size=30) + sum(line([t[0], t[1], t[2], t[0]]) for t in triangulation) + point(point, color='red') + sum(line([t[0], t[1], t[2], t[0]], color='red') for t in bad_triangles)
        show(s, figsize=2.5, axes=False)
        for triangle in bad_triangles:
            triangulation.remove(triangle)
        
        polygon = identifyPolygon(bad_triangles)
        s = point(points, color='black', size=30) + sum(line([t[0], t[1], t[2], t[0]]) for t in triangulation) + point(point, color='red') + sum(line([t[0], t[1], t[0]], color='red') for t in polygon)
        show(s, figsize=2.5, axes=False)
        for edge in polygon:
            triangulation.append((edge[0], edge[1], point))
        s = point(points, color='black', size=30) + sum(line([t[0], t[1], t[2], t[0]]) for t in triangulation) + point(point, color='red')
        show(s, figsize=2.5, axes=False)

    triangulation_iter = triangulation.copy()
    for triangle in triangulation_iter:
        for vertex in triangle:
            if vertex in first_triangle[0]:
                triangulation.remove(triangle)
                break

def delaunay(T, edges):
    """
    T: Triangles from the other algorithm
    edges: edges added from the algorithm in the triangulation
    Returns:
        T: Delaunay Triangles
        edges: new edges
    """
    illegal = True
    n = len(T)
    p = len(edges)
    while illegal:  # While there are illegal edges (flip required)
        illegal = False  # Assume no illegal edges and check for them in a pass
        for i in range(p):
            aux = []  # List of the 4
            for j in range(n):
                if edgeInTriangle(edges[i], T[j]):
                    aux.append(j)
            # Get the 4 unique vertices and the polygon with 4 sides
            adj = getAdjacent(T[aux[0]], T[aux[1]])
            legal, triangles = getLegalEdge(adj, [T[aux[0]], T[aux[1]]])  # Check if it's legal and ensure it doesn't intersect with any edges
            if not compareEdges(legal, edges[i]):  # Compare if the edges are the same
                # Update the triangles and edges
                T[aux[0]] = triangles[0]
                T[aux[1]] = triangles[1]
                edges[i] = legal
                illegal = True  # Set to True to check again
    return T, edges


