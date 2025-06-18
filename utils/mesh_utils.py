import numpy as np
import torch
import copy

import open3d as o3d

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer,
    FoVPerspectiveCameras, look_at_view_transform
)

#===
#pytorch3d mesh
#===
def init_mesh_torch(verts, faces, device='cuda'):
  return Meshes(verts=[verts], faces=[faces]).to(torch.device(device))
  
def sample_mesh_torch(mesh, num_points):
    return sample_points_from_meshes(mesh, num_samples=self.num_points)[0]
    
def render_mesh_torch(
    mesh: Meshes,
    camera_pos: torch.Tensor,
    num_points: int = 1024,
    # image_size: int = 256,
    image_width: int = 320,
    image_height: int = 240,
    max_attempts: int = 3,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    camera_pos.to(device)
    view_vector = - camera_pos
    view_vector = view_vector / (torch.norm(view_vector, dim=-1, keepdim=True) + 1e-8)
    up = torch.tensor([0., 1., 0.], device=device)
    if torch.allclose(torch.abs(torch.dot(view_vector, up)), torch.tensor(1.0, device=view_vector.device), atol=1e-3):
        up = torch.tensor([1., 0., 0.], device=device) # If colinear, switch
    R, T = look_at_view_transform(eye=camera_pos[None], up=up[None], device=device) # z=at-eye, x=cross(up, z), y=cross(z, x)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, aspect_ratio=1.*image_width/image_height) # FoVPerspectiveCameras, OrthographicCameras

    for attempt in range(max_attempts):
        raster_settings = RasterizationSettings(image_size=[image_width, image_height], bin_size=0)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(mesh)

        # Extract render outputs
        zbuf = fragments.zbuf.squeeze(0)[..., 0]  # (H, W)
        pix_to_face = fragments.pix_to_face.squeeze(0)[..., 0]
        bary_coords = fragments.bary_coords.squeeze(0)[..., 0, :]

        # Mask valid points
        valid_mask = zbuf != -1
        valid_faces = pix_to_face[valid_mask]
        valid_bary = bary_coords[valid_mask]

        # No valid points? Retry at higher resolution
        if valid_faces.numel() == 0:
            # image_size = int(image_size * 2)
            image_width = int(image_width * 2)
            image_height = int(image_height * 2)
            continue

        verts = mesh.verts_packed()       
        faces = mesh.faces_packed()[valid_faces]
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        points = valid_bary[:, 0:1] * v0 + valid_bary[:, 1:2] * v1 + valid_bary[:, 2:3] * v2
        
        # If enough points, sample to exactly num_points
        if points.shape[0] >= num_points:
            indices = torch.randperm(points.shape[0], device=device)[:num_points]
            points = points[indices]
        else:
            # Try higher resolution
            # image_size = int(image_size * 2)
            image_width = int(image_width * 2)
            image_height = int(image_height * 2)

    # Fallback if all attempts failed
    if points.shape[0] < num_points:
        pad = num_points - points.shape[0]
        pad_indices = torch.randint(0, points.shape[0], (pad,), device=device)
        points = torch.cat([points, points[pad_indices]], dim=0)
    
    # center = points.mean(dim=0, keepdim=True)           # (1, 3)
    # points = points - center                            # center at origin    
    # scale = torch.norm(points, dim=1).max()             # scalar
    # points = points / scale                             # scale to unit sphere    
    # points[:, [0, 2]] *= -1  
    return points
    
#===
#o3d mesh
#===
def init_mesh(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

def read_mesh(file):
    mesh = o3d.io.read_triangle_mesh(file)
    return mesh

def align_mesh(mesh, class_name):
    '''
    for mn11
    '''
    if class_name in ['bed', 'wardrobe', 'chair', 'door', 'bookshelf', 'sink', 'toilet']:
        mesh = rotate_mesh(mesh, [-np.pi/2, 0, 0])
    elif class_name in ['dresser', 'bench', 'stool', 'desk', 'monitor', 'sofa', 'table']:
        mesh = rotate_mesh(mesh, [-np.pi/2, 0, np.pi/2])
    return mesh

def viz_mesh(meshes, spacing=2):
    '''
    pcds: list of pcd objs; or a single pcd.
    '''
    if not isinstance(meshes, list):
       tmps = [meshes]
    else:
        rows = int(np.sqrt(len(meshes)))
        cols = len(meshes) // rows
        tmps = []
        for idx, mesh in enumerate(meshes):
            row = idx // cols
            col = idx % cols
            translation = np.array([col * spacing, -row * spacing, 0])
            tmp = copy.deepcopy(mesh)
            tmp = tmp.translate(translation)
            tmp.compute_vertex_normals()
            tmps.append(tmp)
    o3d.visualization.draw_geometries(tmps, mesh_show_wireframe=True, mesh_show_back_face=True)    
    
def sample_mesh(mesh, num_points):
    '''
    return pcd obj
    '''
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    # pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    return pcd

def normalize_mesh(mesh):
    origin = [0, 0, 0]
    tmp = copy.deepcopy(mesh)
    tmp = tmp.translate(origin, relative=False)
    vertices = np.asarray(tmp.vertices)
    vertices /= np.max(np.linalg.norm(vertices, axis=1))
    # tmp.vertices = o3d.utility.Vector3dVector(vertices)
    return tmp

def rotate_mesh(mesh, rotation_angles, center=None):
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation_angles)
    tmp = copy.deepcopy(mesh)
    if center is None:
        center = tmp.get_center()
    tmp = tmp.rotate(rotation_matrix, center=center)
    return tmp

def view_z(mesh, res=0.03, num_points=None):
    '''
    view mesh from z-positive.
    '''
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    x = np.arange(min_bound[0], max_bound[0], res)
    y = np.arange(min_bound[1], max_bound[1], res)
    xv, yv = np.meshgrid(x, y)
    z = np.full_like(xv, max_bound[2] + 1.0)  # Points slightly above the mesh
    origins = np.vstack((xv.ravel(), yv.ravel(), z.ravel())).T
    directions = np.array([0, 0, -1], dtype=np.float32).reshape(1, 3) 
    directions = np.tile(directions, (origins.shape[0], 1))
    hits = scene.cast_rays(o3d.core.Tensor(np.hstack((origins, directions)), dtype=o3d.core.Dtype.Float32))
    hit_mask = hits['t_hit'].numpy() < np.inf  # Only consider rays that hit something
    
    # retain visible faces
    # hit_faces = np.unique(hits['primitive_ids'].numpy()[hit_mask])
    # visible_triangles = np.array(mesh.triangles)[hit_faces]
    # visible_vertices = np.array(mesh.vertices)
    # visible_mesh = init_mesh(visible_vertices, visible_triangles)
    # return visible_mesh
    
    # retain first hit points
    hit_points = origins[hit_mask] + hits['t_hit'].numpy()[hit_mask][:, None] * directions[hit_mask]
    if num_points is None:        
        return hit_points
    else:
        if len(hit_points) < num_points:
            res /= 2
            return view_z(mesh, res=res, num_points=num_points)
        else:
            indices = np.random.choice(hit_points.shape[0], num_points, replace=False)
            sampled_points = hit_points[indices]
            return sampled_points

def view_perturb(mesh, rotation_angles):
    mesh = rotate_mesh(mesh, rotation_angles)
    viewed_mesh = view_z(mesh)
    visible_mesh = rotate_mesh(viewed_mesh, -np.array(rotation_angles), center=mesh.get_center())
    return visible_mesh
    
def combine_mesh(meshes):
    combined_vertices = []
    combined_triangles = []
    current_vertex_count = 0
    for mesh in meshes:
        vertices = np.array(mesh.vertices)
        triangles = np.array(mesh.triangles)
        combined_vertices.append(vertices)
        combined_triangles.append(triangles + current_vertex_count)
        current_vertex_count += len(vertices)
    combined_vertices = np.vstack(combined_vertices)
    combined_triangles = np.vstack(combined_triangles)
    combined_mesh = init_mesh(combined_vertices, combined_triangles)
    return combined_mesh
    
def scan(mesh, scanner_pos, res=1/2, num_points=None):
    '''
    simulate real scan with a scanner positioned at 'scanner_pos' that scans with 'res' resolution at angles (azimuth, elevation).
    azimuth is the angle from x-pos-axis to ray direction on xy-plane. [0, 360] in degrees.
    elevation is the vertical angle from xy-plane to ray direction along z-pos-axis. [-90, 90].
    return scanned points
    '''    
    origins = []
    directions = []
    azimuths = np.arange(0, 360, res)
    elevations = np.arange(-90, 90, res)
    for phi in np.radians(azimuths):
        for theta in np.radians(elevations):
            direction = np.array([
                np.cos(theta) * np.cos(phi),
                np.cos(theta) * np.sin(phi),
                np.sin(theta)
            ])
            directions.append(direction)
            origins.append(np.array(scanner_pos))
    origins = np.array(origins)
    directions = np.array(directions)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hits = scene.cast_rays(o3d.core.Tensor(np.hstack((origins, directions)), dtype=o3d.core.Dtype.Float32))
    hit_mask = hits['t_hit'].numpy() < np.inf  # Only consider rays that hit something
    hit_points = origins[hit_mask] + hits['t_hit'].numpy()[hit_mask][:, None] * directions[hit_mask]
    if num_points is None:        
        return hit_points
    else:
        if len(hit_points) < num_points:
            res /= 2
            return scan(mesh, scanner_pos, res=res, num_points=num_points)
        else:
            indices = np.random.choice(hit_points.shape[0], num_points, replace=False)
            sampled_points = hit_points[indices]
            return sampled_points
    
'''
mesh calculations
vertices.shape = (num_vertices, 3)
triangles.shape = (num_triangles, 3)
for each triganle, the 3 values are indices to vertices that make up of the triangle
'''
def compute_triangle_normals(vertices, triangles):   
    '''
    face normal = cross prod of any two crossed lines on this face; in this case, use two edges
    '''
    triangle_normals = []
    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        face_normal /= (np.linalg.norm(face_normal) + 1e-10) # normalization; add a small value to avoid dividend = 0
        triangle_normals.append(face_normal)
    return np.array(triangle_normals)
   
def compute_vertex_normals(vertices, triangles):       
    '''
    vertex normal = sum of normals of 3 facea that are attached to the vertex
    '''
    vertex_normals = np.zeros((len(vertices), 3), dtype=np.float64)
    triangle_normals = compute_triangle_normals(vertices, triangles)
    for i, tri in enumerate(triangles):
        face_normal = triangle_normals[i]    
        # Accumulate face normals to vertices
        vertex_normals[tri[0]] += face_normal
        vertex_normals[tri[1]] += face_normal
        vertex_normals[tri[2]] += face_normal    
    # Normalize vertex normals
    for i in range(len(vertex_normals)):
        vertex_normals[i] /= (np.linalg.norm(vertex_normals[i]) + 1e-10)

def compute_volume(vertices, triangles):
    '''
    input mesh should be translated to origin already.
    '''
    volume = 0.0
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        volume += np.abs(np.dot(v0, np.cross(v1, v2))) / 6 # the volume of a tetrahedron with vertices v0, v1, v2, and the origin
    return volume

'''
mesh sampling
'''
def compute_triangle_areas(vertices, triangles):
    '''
    return areas of each triangle
    '''
    # areas = np.zeros(len(triangles))
    # for i, tri in enumerate(triangles):
    #     v0, v1, v2 = vertices[tri]
    #     areas[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    # return areas    
    areas = torch.zeros(len(triangles), device=vertices.device)
    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri]
        areas[i] = 0.5 * torch.norm(torch.linalg.cross(v1 - v0, v2 - v0))
    return areas

def uniformly_sample_triangle(vertices):
    '''
    sample one point from a triangle.
    vertices = [v0, v1, v2] of the triangle.
    '''
    # v0, v1, v2 = vertices
    # u, v = np.random.rand(2)
    # sample = (1 - np.sqrt(u)) * v0 + np.sqrt(u) * (1 - v) * v1 + np.sqrt(u) * v * v2
    # return sample    
    v0, v1, v2 = vertices
    u, v = torch.rand(2, device=vertices.device)
    sample = (1 - torch.sqrt(u)) * v0 + torch.sqrt(u) * (1 - v) * v1 + torch.sqrt(u) * v * v2
    return sample

def uniformly_sample(vertices, triangles, n_samples):
    # areas = compute_triangle_areas(vertices, triangles)
    # total_area = np.sum(areas)
    # probabilities = areas / total_area
    # sampled_triangles = np.random.choice(len(triangles), size=n_samples, p=probabilities)
    # points = [uniformly_sample_triangle(vertices[triangles[tri_idx]]) for tri_idx in sampled_triangles]
    # return np.array(points)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    triangles = torch.tensor(triangles, dtype=torch.int64, device=device)
    areas = compute_triangle_areas(vertices, triangles)
    total_area = torch.sum(areas)
    probabilities = areas / total_area
    sampled_triangles = torch.multinomial(probabilities, n_samples, replacement=True)
    points = [uniformly_sample_triangle(vertices[triangles[tri_idx]]) for tri_idx in sampled_triangles]
    return torch.stack(points).cpu().numpy()

'''
pcl/tools/mesh_sampling.cpp
'''
def uniform_deviate(seed):
    return seed * (1.0 / (np.iinfo(np.int32).max + 1.0))

def random_point_triangle(a, b, c, r1, r2):
    r1_sqrt = np.sqrt(r1)
    one_minus_r1_sqrt = (1 - r1_sqrt)
    one_minus_r2 = (1 - r2)
    point = a * one_minus_r1_sqrt + b * one_minus_r2 * r1_sqrt + c * r2 * r1_sqrt
    return point

def rand_p_surface(vertices, triangles, cumulative_areas, total_area):
    r = uniform_deviate(np.random.randint(0, np.iinfo(np.int32).max)) * total_area
    el = np.searchsorted(cumulative_areas, r)
    triangle = triangles[el]
    a = vertices[triangle[0]]
    b = vertices[triangle[1]]
    c = vertices[triangle[2]]
    r1 = uniform_deviate(np.random.randint(0, np.iinfo(np.int32).max))
    r2 = uniform_deviate(np.random.randint(0, np.iinfo(np.int32).max))
    return random_point_triangle(a, b, c, r1, r2)

def uniform_sampling(vertices, triangles, n_samples):
    areas = np.zeros((triangles.shape[0]))
    
    for i, triangle in enumerate(triangles):
        a = vertices[triangle[0]]
        b = vertices[triangle[1]]
        c = vertices[triangle[2]]
        areas[i] = np.linalg.norm(np.cross(b - a, c - a)) / 2
    
    total_area = np.sum(areas)
    cumulative_areas = np.cumsum(areas)
    points = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        points[i] = rand_p_surface(vertices, triangles, cumulative_areas, total_area)
    
    return points