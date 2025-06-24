import copy
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def normalize_pcd_tensor(points):
    center = points.mean(dim=0, keepdim=True)           # (1, 3)
    points = points - center                            # center at origin    
    scale = torch.norm(points, dim=1).max()             # scalar
    points = points / scale                             # scale to unit sphere    
    return points   
    
#=====
#pointnet++: fps-ball-group
#=====
def ball_query(centers, points, radius, nsample):
    B, G, _ = centers.shape
    N = points.shape[1]

    # Compute squared distance matrix (B, G, N)
    dist = torch.cdist(centers, points, p=2)  # (B, G, N)

    # Mask out distances greater than radius
    mask = dist <= radius
    idx = mask.nonzero(as_tuple=False)  # (num_valid, 3): [B_idx, G_idx, N_idx]

    # Initialize with default index (e.g., nearest or zero)
    output_idx = torch.full((B, G, nsample), fill_value=0, device=points.device, dtype=torch.long)

    for b in range(B):
        for g in range(G):
            valid = torch.nonzero(mask[b, g], as_tuple=False).squeeze(-1)
            if valid.numel() >= nsample:
                output_idx[b, g] = valid[:nsample]
            else:
                # pad by repeating first neighbor
                pad = valid[0].repeat(nsample - valid.numel())
                output_idx[b, g] = torch.cat([valid, pad], dim=0)
    return output_idx

def sample_and_group_ball(points, num_group, group_size, radius):
    '''
    points: (B, N, 3)
    returns:
      - neighborhoods: (B, G, M, 3)
      - centers: (B, G, 3)
    '''
    centers = farthest_point_sample_gpu_batch(points, num_group)  # (B, G, 3)
    idx = ball_query(centers, points, radius, group_size)         # (B, G, M)
    neighborhoods = group_points(points, idx)                     # (B, G, M, 3)
    neighborhoods = neighborhoods - centers.unsqueeze(2)          # center relative
    return neighborhoods, centers


#=====
#pointmae: fps-knn-group
#=====
def farthest_point_sample_gpu_batch(points, n):
    '''
    points: (B, N, 3) torch.Tensor
    return: (B, n, 3)
    '''
    B, N, _ = points.shape
    device = points.device

    centroids = torch.zeros(B, n, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(n):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest]  # (B, 3)
        dist = torch.sum((points - centroid[:, None, :]) ** 2, dim=-1)  # (B, N)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, dim=1)[1]

    sampled_points = points[batch_indices[:, None], centroids]  # (B, n, 3)
    return sampled_points

def knn_group(points, centers, k):
    '''
    points: (B, N, 3)
    centers: (B, G, 3)
    return: (B, G, k, 3)
    '''
    B, N, C = points.shape
    G = centers.shape[1]

    dists = torch.cdist(centers, points)  # (B, G, N)
    idx = dists.topk(k, dim=-1, largest=False)[1]  # (B, G, k)

    idx_base = torch.arange(B, device=points.device).view(B, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    grouped = points.reshape(B * N, C)[idx, :].view(B, G, K, C)
    return grouped


def group_points(points, idx):
    '''
    points: (B, N, 3)
    idx: (B, G, k)
    return: (B, G, k, 3)
    '''
    B, N, C = points.shape
    B, G, K = idx.shape

    idx_base = torch.arange(B, device=points.device).view(B, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    grouped = points.reshape(B * N, C)[idx, :].view(B, G, K, C)
    return grouped

def sample_and_group(points, num_group, group_size):
    '''
    points: (B, N, 3)
    returns:
      - neighborhoods: (B, G, M, 3)
      - centers: (B, G, 3)
    '''
    centers = farthest_point_sample_gpu_batch(points, num_group)  # (B, G, 3)
    idx = knn_gather(points, centers, group_size)  # (B, G, M)
    neighborhoods = group_points(points, idx)      # (B, G, M, 3)
    neighborhoods = neighborhoods - centers.unsqueeze(2)
    return neighborhoods, centers

#=====
#my funcs
#=====
def knneigval(points, k=32):
    N = points.shape[0]
    eigvals = np.zeros(N)
    # covariances = np.zeros((N, 3, 3))  # Store covariance matrices
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    _, indices = nbrs.kneighbors(points)
    for i in range(N):
        neighbors = points[indices[i]]  # Get k nearest neighbors
        mean_neighbor = np.mean(neighbors, axis=0)  # Compute centroid
        centered_neighbors = neighbors - mean_neighbor  # Center the neighbors
        cov_matrix = (centered_neighbors.T @ centered_neighbors) / k
        # covariances[i] = cov_matrix
        eigvals[i] = np.linalg.eigh(cov_matrix)[0][-1] # only store largest eigval
    return eigvals

def furthestX(points):
    distances = np.linalg.norm(points[:, [0, 2]], axis=1)
    furthest_idx = np.argmax(distances)
    p_far = points[furthest_idx]    
    theta = np.arctan2(p_far[2], p_far[0])
    return rotate_xyz(points, [0, theta, 0])

def covbasis(points):
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    projections = points @ eigenvectors
    projection_lengths = np.max(projections, axis=0) - np.min(projections, axis=0)
    sorted_indices = np.argsort(projection_lengths)
    eigenvectors = eigenvectors[:, sorted_indices]
    projections = points @ eigenvectors
    for i in range(eigenvectors.shape[1]):
        positive_side = np.sum(projections[:, i] > 0)
        negative_side = np.sum(projections[:, i] < 0)
        if negative_side > positive_side:  # If more points are on the negative side, flip the direction
            eigenvectors[:, i] *= -1
    return points @ eigenvectors

def cart2cy(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x**2 + z**2)  # Radial distance
    # y_normalized = (y - min(y)) / (max(y) - min(y))
    # y_normalized = (y + 1) / 2
    xmin = x[np.argmin(y)]
    zmin = z[np.argmin(y)]
    theta = np.arctan2(xmin * z - zmin * x, xmin * x + zmin * z) # range: [-p, p]
    # theta_normalized = (theta + np.pi) / (2 * np.pi)    
    theta_unwrapped = np.where(theta < 0, theta + 2 * np.pi, theta)  # [0, 2p]
    theta_normalized = theta_unwrapped / (2 * np.pi)  # final range: [0, 1]
    return np.column_stack((r, theta_normalized, y))

def compute_matrices(points):
    '''
    return dist_matrix, angle_matrix.
    dist_matrix: dist of p1-p2.
    angle_matrix: radian angle of p1-O-p2. [0, pi].
    symmetric.
    '''
    dot_product_matrix = np.dot(points, points.T)
    magnitudes = np.linalg.norm(points, axis=1)
    magnitude_matrix = np.outer(magnitudes, magnitudes)
    cos_theta_matrix = dot_product_matrix / magnitude_matrix
    cos_theta_matrix = np.clip(cos_theta_matrix, -1.0, 1.0)
    angle_matrix = np.arccos(cos_theta_matrix)
    return cdist(points, points), angle_matrix

def scale0(label, points):
    '''
    scale_factor defined by hongtao.
    implement after normalization.
    '''
    if label == 0 or label == 6:
        points *= 2
    elif label == 1:
        points *= 2.5
    elif label == 2 or label == 10:
        points *= 0.5
    elif label == 4 or label == 7:
        points *= 0.4
    return points

def get_bound(points):
    '''
    return bound=(3,).
    '''
    return np.max(points, axis=0) - np.min(points, axis=0)

def centralize(points):
    '''
    move centroid to origin
    '''
    centroid = np.mean(points, axis=0)
    points -= centroid
    return points

def get_furthest_dist(points):
    points = centralize(points)
    return np.max(np.linalg.norm(points, axis=1))

def fill(points, num_points, filler_point=None):
    '''
    set len(points) to num_points.
    if too long, downsample.
    if too short, fill with filler_point or first point.
    '''
    if len(points) > num_points:
        points = uniformly_sample(points, num_points)
    else:
        if filler_point is None:
            filler_point = points[0]
        points = np.vstack((points, np.tile(filler_point, (num_points - len(points), 1))))
    return points

# ===========================
# o3d pcd obj
# ===========================
def init_pcd(points, **kwargs):
    '''
    kwargs: normals=(num_points, 3), colors(in rgb)=(3, ) or (num_points, 3).
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if 'normals' in kwargs:
        normals = np.array(kwargs['normals'])
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if 'colors' in kwargs:
        colors = np.array(kwargs['colors'])
        if colors.shape != points.shape:
            colors = np.tile(colors, (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors / 225.0)   
    return pcd

def viz_pcd(pcds, spacing=2, rows=None):
    '''
    pcds: list of pcd objs; or a single pcd.
    '''
    if not isinstance(pcds, list):
        pcds = [pcds]
    if rows is None:
        rows = int(np.sqrt(len(pcds)))
    cols = len(pcds) // rows
    tmps = []
    for idx, pcd in enumerate(pcds):
        if not isinstance(pcd, o3d.geometry.PointCloud):
            pcd = init_pcd(pcd)
        row = idx // cols
        col = idx % cols
        translation = np.array([col * spacing, -row * spacing, 0])
        tmp = copy.deepcopy(pcd)
        tmp = tmp.translate(translation)
        tmps.append(tmp)
    o3d.visualization.draw_geometries(tmps)
    
def normalize_pcd(pcd):
    '''
    input/output pcd = o3d pointcloud obj.
    '''
    origin = [0, 0, 0]
    tmp = copy.deepcopy(pcd)    
    tmp = tmp.translate(origin, relative=False)
    tmp = tmp.scale(1./max(tmp.compute_point_cloud_distance(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector([origin])))), center=origin)
    return tmp

def rotate_pcd(pcd, rotation_angles):
    '''
    input/output pcd = o3d pointcloud obj.
    rotation_angles in radian (e.g., np.pi/2).
    rotate around pcd.center.
    '''
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation_angles)
    tmp = copy.deepcopy(pcd)
    tmp = tmp.rotate(rotation_matrix, center=tmp.get_center())
    return tmp

'''
pcd transformation
consider right-handed coordinate system (used in 3D graphics)
points.shape = (num_points, dimension=3 or 6 if with normals)
'''
def normalize(points):
    '''
    normalize into a unit ball
    '''
    centroid = np.mean(points, axis=0)
    points -= centroid # translation, move centroid to origin
    furthest_distance = np.max(np.linalg.norm(points, axis=1)) # max distance from points to origin (centroid)
    points /= furthest_distance # scaling
    return points

def rotate_xyz(points, rotation_angles):
    '''    
    input points should be centered at origin.
    rotation_angles in radian (e.g., np.pi/2).
    positive rotatation_angle follows the right-hand rule.
    X-axis: Points horizontally from left to right (on the screen).
    Y-axis: Points vertically from bottom to top (on the screen).
    Z-axis: Points out of the screen (from the back) towards the viewer.
    '''
    theta_x, theta_y, theta_z = rotation_angles
      
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x

    if isinstance(points, torch.Tensor):
        R = torch.tensor(R, dtype=points.dtype, device=points.device)

    return points @ R.T

def generate_viewpoints(num_views, radius=2.0):
    '''
    generate uniformly scattered points within a spherical shell (defined by radius) around the unit sphere
    '''
    theta = np.random.uniform(0, 2 * np.pi, num_views) # azimuth angle determines the position around the equator of the unit sphere
    phi = np.random.uniform(0, np.pi, num_views)  # elevation angle determines the angle above or below the equator
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack((x, y, z), axis=-1)

def view_z(points, n):
    '''
    input points should be normalized.
    view from z-positive axis, occlude points cannot be seen.
    downsample to n points.    
    '''
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, _ = np.max(points, axis=0)    
    cell_size = 0.035
    while 1:    
        dic = {} # (grid_idx: indices of points within this grid)
        for i, (x, y, _) in enumerate(points):
            u = int((x - min_x) / cell_size)
            v = int((y - min_y) / cell_size)        
            if (u, v) in dic:
                dic[(u, v)] = np.append(dic[(u, v)], i)
            else:
                dic[(u, v)] = np.array([i])
        for key in dic.keys():
            d = points[:, 2][dic[key]] - min_z
            # dic[key] = dic[key][d >= max(d) - 0.1] # (grid_idx: selected points indices)
            dic[key] = dic[key][d >= max(d) * 0.9]
        visible_indices = np.concatenate(list(dic.values()))
        if len(visible_indices) < n:
            cell_size -= 0.003
        # elif len(visible_indices) > 1.8 * n:
        #     cell_size += 0.003
        else:
            visible_indices = visible_indices[np.random.choice(len(visible_indices), n, replace=False)]
            break
    return points[visible_indices]

def dropout(points, dropout_ratio):
    num_points = points.shape[0]
    dropout_idx = np.random.choice(num_points, size=int(num_points*dropout_ratio), replace=False)
    # dropout_idx = np.where(np.random.random(points.shape[0]) <= dropout_ratio)[0] # points are selected for dropout by generating a random number for each point and checking if it is less than or equal to the dropout ratio.
    # dropout_idx = np.unique(np.concatenate((np.random.choice(num_points, size=int(num_points*dropout_ratio), replace=False),
    #                                        np.where(np.random.random(num_points) <= np.random.random() * 0.0875)[0])))
    if len(dropout_idx) > 0:
        points[dropout_idx] = points[0] # Set dropped points to the first point
        # points[dropout_idx] = points[np.random.choice(points.shape[0], size=dropout_idx.shape, replace=True)] # Set dropped points to a random point
    return points

'''
used in pointnet2
'''
def random_drop_out(points, max_dropout_ratio=0.875):
    '''
    dropout points with random dropout ratio with max limit.
    dropped points are refilled with the first point
    '''
    if isinstance(points, np.ndarray):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0 to max_dropout_ratio
        dropout_idx = np.where(np.random.random(points.shape[0]) <= dropout_ratio)[0] # points are selected for dropout by generating a random number for each point and checking if it is less than or equal to the dropout ratio.
        if len(dropout_idx) > 0:
            points[dropout_idx] = points[0] # Set dropped points to the first point
            # points[dropout_idx] = points[np.random.choice(points.shape[0], size=dropout_idx.shape, replace=True)] # Set dropped points to a random point
    else:
        B = points.shape[0] if points.dim() == 2 else 1
        dropout_ratio = torch.rand(1, device=points.device).item() * max_dropout_ratio
        mask = torch.rand(points.shape[0], device=points.device) <= dropout_ratio
        if mask.any():
            points[mask] = points[0].clone()
    return points

def random_scale(points, scale_low=0.8, scale_high=1.25):
    scale = np.random.uniform(scale_low, scale_high) if isinstance(points, np.ndarray) else \
            torch.empty(1, device=points.device).uniform_(scale_low, scale_high).item()
    return points * scale

def random_shift(points, shift_range=0.1):
    '''
    translation
    '''
    if isinstance(points, np.ndarray):
        shift = np.random.uniform(-shift_range, shift_range, 3)
    else:
        shift = torch.empty(3, device=points.device).uniform_(-shift_range, shift_range)
    return points + shift

def rotate_perturb(points, sigma=0.06, clip=0.18): # 0.05 rad = 3 degree
    '''
    add perturbation to rotation around three axes.
    limit rotation angle to -clip to clip with std=sigma.
    '''
    if isinstance(points, np.ndarray):
        angles = np.clip(sigma * np.random.randn(3), -clip, clip)
    else:
        angles = torch.clamp(sigma * torch.randn(3, device=points.device), -clip, clip)
    return rotate_xyz(points, angles)

def jitter(points, std=0.01, clip=0.05):
    '''
    add gaussian noise but keep noise level within -clip to clip.
    std is sometimes called sigma.
    '''
    if isinstance(points, np.ndarray):
        noise = np.clip(np.random.normal(0, std, points.shape), -clip, clip)
    else:
        noise = torch.clamp(torch.randn_like(points) * std, -clip, clip)
    return points + noise

def random_rotate3d(points):
    if isinstance(points, np.ndarray):
        angles = np.random.rand(3) * 2 * np.pi
    else:
        angles = torch.rand(3, device=points.device) * 2 * np.pi
    return rotate_xyz(points, angles)

'''
used in dgcnn
'''
def translate(points):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])       
    translated_pointcloud = np.add(np.multiply(points, xyz1), xyz2).astype('float32')
    return translated_pointcloud

'''
used in metaset
'''
def drop_hole(pc, p):
    '''
    remove 'p' percentage of points closest to a random point
    '''
    random_point = np.random.randint(0, pc.shape[0])
    index = np.linalg.norm(pc - pc[random_point].reshape(1,3), axis=1).argsort() # indices of points sorted by dist to random_point selected
    return pc[index[int(pc.shape[0] * p):]]
    
def density(pc, v_point=np.array([1, 0, 0]), gate=1):
    '''
    downsample based on proximity to v_point, but with randomness.
    '''
    dist = np.sqrt((v_point ** 2).sum()) # dist from v_point to origin
    max_dist = dist + 1
    min_dist = dist - 1
    dist = np.linalg.norm(pc - v_point.reshape(1,3), axis=1) # dist from v_point to each point in pcd
    dist = (dist - min_dist) / (max_dist - min_dist) # normalize to [0, 1]
    r_list = np.random.uniform(0, 1, pc.shape[0]) # random gating
    tmp_pc = pc[dist * gate < (r_list)] # retain point if scaled dist (by 'gate') to v_point < r_list
    return tmp_pc

def p_scan(pc, pixel_size=0.017):
    '''
    project pcd onto a 2D grid
    '''
    pixel = int(2 / pixel_size) # num of pixels along one axis (-1 to 1)
    rotated_pc = random_rotate3d(pc)
    pc_compress = (rotated_pc[:,2] + 1) / 2 * pixel * pixel + (rotated_pc[:,1] + 1) / 2 * pixel # unique grid index for each point; (z, y coordinates + 1)/2 normalizes coor from range [-1, 1] to [0, 1]
    points_list = [None for i in range((pixel + 5) * (pixel + 5))] # indices of points; slightly larger than pixel*pixel to accommodate any potential overflow
    pc_compress = pc_compress.astype(int)
    for index, point in enumerate(rotated_pc):
        compress_index = pc_compress[index]
        if compress_index > len(points_list):
            print('out of index:', compress_index, len(points_list), point, pc[index], (pc[index] ** 2).sum(), (point ** 2).sum())
        if points_list[compress_index] is None:
            points_list[compress_index] = index # select the point
        elif point[0] > rotated_pc[points_list[compress_index]][0]: # x coor bigger than already selected
            points_list[compress_index] = index
    points_list = list(filter(lambda x:x is not None, points_list)) # filter out None values
    points_list = pc[points_list]
    return points_list

'''
pcd sampling
points.shape = (num_points, 3)
n = num_sampled_points
'''
def uniformly_sample(points, n):
    indices = np.random.choice(points.shape[0], n, replace=False)
    sampled_points = points[indices]
    return sampled_points

def farthest_point_sample_o3d(points, n):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.farthest_point_down_sample(n)
    return np.array(pcd.points)
    
def farthest_point_sample(points, n):
    '''
    FPS: iteratively selects the point that is farthest from the previously selected points.
    preserves the distribution of points (bettter than uniform sampling)
    '''
    num_points = points.shape[0]
    sampled_indices = np.zeros(n, dtype=np.int32) # indices of the sampled points in the original point cloud
    distances = np.full(num_points, np.inf) # min dist of each point in the original pcd from sampled points; initialized to a very big value
    farthest_idx = np.random.randint(num_points) # idx of farthest point; randomly select the starting point

    for i in range(n):
        sampled_indices[i] = farthest_idx # add farthest point to sampled point set
        farthest_point = points[farthest_idx]
        dist = np.sum((points - farthest_point) ** 2, axis=1) # squared Enclidean dist of each point in original pcd from farthest point
        distances = np.minimum(distances, dist) # dist to the nearest sampled point
        farthest_idx = np.argmax(distances)

    sampled_points = points[sampled_indices]
    return sampled_points

def farthest_point_sample_gpu(points, n):
    '''
    FPS: iteratively selects the point that is farthest from the previously selected points.
    preserves the distribution of points (bettter than uniform sampling).
    return tensor.
    in torch, use gpu
    '''
    device = points.device
    num_points = points.shape[0]
    sampled_indices = torch.zeros(n, dtype=torch.long, device=device)
    distances = torch.full((num_points,), float('inf'), device=device)
    farthest_idx = torch.randint(0, num_points, (1,), device=device).item()

    for i in range(n):
        sampled_indices[i] = farthest_idx
        farthest_point = points[farthest_idx]
        dist = torch.sum((points - farthest_point) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        farthest_idx = torch.argmax(distances).item()

    sampled_points = points[sampled_indices]
    return sampled_points

def density_based_sample(points, n):
    '''
    higher density, less importance
    '''
    num_points = len(points)
    densities = np.zeros(num_points)

    # Compute the density of each point (number of neighbors within a certain radius)
    for i, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        densities[i] = np.sum(distances < np.percentile(distances, 5))

    # Inverse of densities to make high density less important
    importance = 1 / (densities + 1e-5)  # Add small value to avoid division by zero

    # Normalize importance weights
    importance /= np.sum(importance)

    # Sample points based on the computed importance
    sampled_indices = np.random.choice(num_points, n, p=importance)
    sampled_points = points[sampled_indices]
    
    return sampled_points

