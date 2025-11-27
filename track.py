import pygame
import numpy as np

def catmull_rom_spline_oriented(P0, P1, P2, P3, n_points=30):
    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    a = -0.5 * P0[:2] + 1.5 * P1[:2] - 1.5 * P2[:2] + 0.5 * P3[:2]
    b = P0[:2] - 2.5 * P1[:2] + 2 * P2[:2] - 0.5 * P3[:2]
    c = -0.5 * P0[:2] + 0.5 * P2[:2]
    d = P1[:2]
    pos = a * t**3 + b * t**2 + c * t + d

    def shortest_angle(a1, a2): 
        return a1 + ((a2 - a1 + 180) % 360 - 180)

    rot1 = shortest_angle(P0[2], P1[2])
    rot2 = shortest_angle(rot1, P2[2])
    rot3 = shortest_angle(rot2, P3[2])

    r_a = -0.5 * P0[2] + 1.5 * rot1 - 1.5 * rot2 + 0.5 * rot3
    r_b = P0[2] - 2.5 * rot1 + 2 * rot2 - 0.5 * rot3
    r_c = -0.5 * P0[2] + 0.5 * rot2
    r_d = rot1

    rot = r_a * t**3 + r_b * t**2 + r_c * t + r_d
    return np.hstack([pos, rot])

def generate_oriented_spline(control_points, n_points_per_segment=30):
    spline_points = []
    n = len(control_points)
    for i in range(n):
        P0, P1, P2, P3 = control_points[i-1], control_points[i], control_points[(i+1) % n], control_points[(i+2) % n]
        segment_points = catmull_rom_spline_oriented(P0, P1, P2, P3, n_points_per_segment)
        spline_points.append(segment_points)
    return np.vstack(spline_points)

def interpolate_widths_along_spline(control_points, n_points_per_segment):
    n_control_points = len(control_points)
    width_per_spline_point = []
    for i in range(n_control_points):
        p1 = control_points[i]
        p2 = control_points[(i + 1) % n_control_points]
        segment_widths = np.linspace(p1[3], p2[3], n_points_per_segment)
        width_per_spline_point.extend(segment_widths)
    return np.array(width_per_spline_point)

def get_track_edges(spline_points, control_points, n_points_per_segment=30):
    left_edge, right_edge = [], []
    width_per_spline_point = interpolate_widths_along_spline(control_points, n_points_per_segment)
    for i, (x, y, angle) in enumerate(spline_points):
        width = width_per_spline_point[i]
        rad = np.radians(angle)
        perp = np.array([np.sin(rad), np.cos(rad)]) * (width / 2)
        left_edge.append([x + perp[0], y + perp[1]])
        right_edge.append([x - perp[0], y - perp[1]])
    return np.array(left_edge), np.array(right_edge)

def _closest_point_on_segment(p, a, b):
    ab = b - a
    ab_len2 = np.sum(ab * ab, axis=-1)
    ab_len2_safe = np.where(ab_len2 == 0, 1e-12, ab_len2)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / ab_len2_safe, 0.0, 1.0)
    closest = a + (ab.T * t).T
    return closest, t

def _segment_normal(a, b):
    dir_vec = b - a
    length = np.linalg.norm(dir_vec, axis=-1, keepdims=True)
    length_safe = np.where(length == 0, 1e-12, length)
    normal = np.stack([-dir_vec[...,1], dir_vec[...,0]], axis=-1) / length_safe
    return normal

def vectorized_collision(car_pos, radius, track_edges_left, track_edges_right):
    car_pos = np.asarray(car_pos, dtype=float)
    N = car_pos.shape[0]
    edges = [track_edges_left, track_edges_right]
    best_dist2 = np.full(N, np.inf)
    best_normals = np.zeros((N, 2), dtype=float)
    best_penetration = np.zeros(N, dtype=float)
    best_cp = np.zeros((N, 2), dtype=float)

    for edge in edges:
        p0 = np.array(edge[:-1])
        p1 = np.array(edge[1:])
        seg = p1 - p0
        seg_len2 = np.sum(seg**2, axis=1)
        seg_len2_safe = np.where(seg_len2==0, 1e-12, seg_len2)
        rel = car_pos[:, None, :] - p0[None, :, :]
        t = np.sum(rel * seg[None, :, :], axis=2) / seg_len2_safe[None, :]
        t_clamped = np.clip(t, 0, 1)
        closest = p0[None, :, :] + seg[None, :, :] * t_clamped[:, :, None]
        diff = car_pos[:, None, :] - closest
        dist2 = np.sum(diff**2, axis=2)
        idx = np.argmin(dist2, axis=1)
        mask = dist2[np.arange(N), idx] < best_dist2
        best_dist2[mask] = dist2[np.arange(N), idx][mask]
        best_cp[mask] = closest[np.arange(N), idx][mask]
        vec = car_pos - best_cp
        dist = np.sqrt(best_dist2)
        normals = vec / np.maximum(dist[:, None], 1e-8)
        best_normals[mask] = normals[mask]

    penetration = radius - np.sqrt(best_dist2)
    collision_mask = penetration > 0
    return collision_mask, best_normals, penetration, best_cp

def compute_rays(car_pos, car_angle, N_rays=5, ray_length=1500.0, ray_fan_deg=60.0):
    N = car_pos.shape[0]
    angles = np.deg2rad(car_angle)
    fan_offsets = np.linspace(-ray_fan_deg/2, ray_fan_deg/2, N_rays)
    fan_offsets_rad = np.deg2rad(fan_offsets)
    total_angles = angles[:, None] + fan_offsets_rad[None, :]
    directions = np.stack([np.cos(total_angles), np.sin(total_angles)], axis=2)
    rays_start = np.repeat(car_pos[:, None, :], N_rays, axis=1)
    rays_end = rays_start + directions * ray_length
    return rays_start, rays_end

def vectorized_ray_cast(rays_start, rays_end, track_edges_left, track_edges_right):
    N,R,_ = rays_start.shape
    edges = [track_edges_left, track_edges_right]
    hit_dist_final = np.full((N,R), np.inf)
    hit_point_final = np.zeros((N,R,2), dtype=float)

    for edge in edges:
        P0 = np.array(edge[:-1])
        P1 = np.array(edge[1:])
        SE = P1 - P0
        S = rays_start[:, :, None, :]
        D = rays_end - rays_start
        D = D[:, :, None, :]
        Q = P0[None,None,:,:]
        SEG = SE[None,None,:,:]
        D_cross_SEG = D[...,0]*SEG[...,1] - D[...,1]*SEG[...,0]
        Q_minus_S = Q - S
        QMS_cross_SEG = Q_minus_S[...,0]*SEG[...,1] - Q_minus_S[...,1]*SEG[...,0]
        QMS_cross_D   = Q_minus_S[...,0]*D[...,1] - Q_minus_S[...,1]*D[...,0]
        denom = np.where(np.abs(D_cross_SEG)<1e-9, 1e-9, D_cross_SEG)
        t = QMS_cross_SEG / denom
        u = QMS_cross_D / denom
        valid = (t>=0) & (u>=0) & (u<=1)
        t_valid = np.where(valid, t, np.inf)
        t_min = np.min(t_valid, axis=2)
        idx_min = np.argmin(t_valid, axis=2)
        hit_pts = S[:,:,0,:] + D[:,:,0,:] * t_min[:,:,None]
        mask = t_min < hit_dist_final
        hit_dist_final[mask] = t_min[mask]
        hit_point_final[mask] = hit_pts[mask]

    hit_mask = hit_dist_final < np.inf
    return hit_mask, hit_dist_final, hit_point_final
