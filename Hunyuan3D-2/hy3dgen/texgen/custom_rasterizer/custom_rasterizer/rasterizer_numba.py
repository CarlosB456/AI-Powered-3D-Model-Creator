"""
Numba-accelerated CPU rasterizer replacement for custom_rasterizer_kernel.
Provides 100% mathematical parity with Tencent's rasterize_image without requiring
NVIDIA CUDA, nvcc, or MSVC compilers.
Compatible with Windows, Linux, and macOS on AMD, Intel, and NVIDIA hardware.
"""
import numpy as np
import numba
import torch

MAXINT = np.int64(2147483647)
MAX_TOKEN = MAXINT * MAXINT + (MAXINT - 1)

@numba.njit(fastmath=True, inline='always')
def calculate_signed_area2(ax, ay, bx, by, cx, cy):
    return (cx - ax) * (by - ay) - (bx - ax) * (cy - ay)

@numba.njit(fastmath=True, inline='always')
def calc_barycentric(vt0, vt1, vt2, px, py):
    area = calculate_signed_area2(vt0[0], vt0[1], vt1[0], vt1[1], vt2[0], vt2[1])
    if area == 0.0:
        return -1.0, -1.0, -1.0
    tri_inv = 1.0 / area
    beta = calculate_signed_area2(vt0[0], vt0[1], px, py, vt2[0], vt2[1]) * tri_inv
    gamma = calculate_signed_area2(vt0[0], vt0[1], vt1[0], vt1[1], px, py) * tri_inv
    alpha = 1.0 - beta - gamma
    return alpha, beta, gamma

@numba.njit(fastmath=True)
def _rasterize_triangles(V, F, width, height, zbuffer, d, occlusion_truncation, has_depth):
    num_faces = F.shape[0]
    w_m1 = float(width - 1)
    h_m1 = float(height - 1)

    for f in range(num_faces):
        idx0 = F[f, 0]
        idx1 = F[f, 1]
        idx2 = F[f, 2]

        w0 = V[idx0, 3]
        w1 = V[idx1, 3]
        w2 = V[idx2, 3]

        vt0_x = (V[idx0, 0] / w0 * 0.5 + 0.5) * w_m1 + 0.5
        vt0_y = (0.5 + 0.5 * V[idx0, 1] / w0) * h_m1 + 0.5
        vt0_z = V[idx0, 2] / w0 * 0.49999 + 0.5

        vt1_x = (V[idx1, 0] / w1 * 0.5 + 0.5) * w_m1 + 0.5
        vt1_y = (0.5 + 0.5 * V[idx1, 1] / w1) * h_m1 + 0.5
        vt1_z = V[idx1, 2] / w1 * 0.49999 + 0.5

        vt2_x = (V[idx2, 0] / w2 * 0.5 + 0.5) * w_m1 + 0.5
        vt2_y = (0.5 + 0.5 * V[idx2, 1] / w2) * h_m1 + 0.5
        vt2_z = V[idx2, 2] / w2 * 0.49999 + 0.5

        vt0 = (vt0_x, vt0_y, vt0_z)
        vt1 = (vt1_x, vt1_y, vt1_z)
        vt2 = (vt2_x, vt2_y, vt2_z)

        x_min = int(max(0.0, min(vt0_x, vt1_x, vt2_x)))
        x_max = int(min(float(width - 1), max(vt0_x, vt1_x, vt2_x)))
        y_min = int(max(0.0, min(vt0_y, vt1_y, vt2_y)))
        y_max = int(min(float(height - 1), max(vt0_y, vt1_y, vt2_y)))

        face_token_id = np.int64(f + 1)

        for py in range(y_min, y_max + 1):
            py_f = py + 0.5
            for px in range(x_min, x_max + 1):
                px_f = px + 0.5
                alpha, beta, gamma = calc_barycentric(vt0, vt1, vt2, px_f, py_f)
                if alpha >= 0.0 and beta >= 0.0 and gamma >= 0.0 and alpha <= 1.0 and beta <= 1.0 and gamma <= 1.0:
                    depth = alpha * vt0_z + beta * vt1_z + gamma * vt2_z
                    if has_depth:
                        depth_thres = d[py, px] * 0.49999 + 0.5 + occlusion_truncation
                        if depth < depth_thres:
                            continue
                    z_quantize = np.int64(depth * (2 << 17))
                    token = z_quantize * MAXINT + face_token_id
                    if token < zbuffer[py, px]:
                        zbuffer[py, px] = token

@numba.njit(parallel=True, fastmath=True)
def _compute_barycentrics(V, F, zbuffer, width, height, findices, barycentric_map):
    w_m1 = float(width - 1)
    h_m1 = float(height - 1)

    for py in numba.prange(height):
        py_f = py + 0.5
        for px in range(width):
            token = zbuffer[py, px]
            f_val = token % MAXINT
            if f_val == (MAXINT - 1) or f_val == 0:
                findices[py, px] = 0
                barycentric_map[py, px, 0] = 0.0
                barycentric_map[py, px, 1] = 0.0
                barycentric_map[py, px, 2] = 0.0
            else:
                findices[py, px] = int(f_val)
                f_idx = int(f_val - 1)

                idx0 = F[f_idx, 0]
                idx1 = F[f_idx, 1]
                idx2 = F[f_idx, 2]

                w0 = V[idx0, 3]
                w1 = V[idx1, 3]
                w2 = V[idx2, 3]

                vt0_x = (V[idx0, 0] / w0 * 0.5 + 0.5) * w_m1 + 0.5
                vt0_y = (0.5 + 0.5 * V[idx0, 1] / w0) * h_m1 + 0.5
                vt1_x = (V[idx1, 0] / w1 * 0.5 + 0.5) * w_m1 + 0.5
                vt1_y = (0.5 + 0.5 * V[idx1, 1] / w1) * h_m1 + 0.5
                vt2_x = (V[idx2, 0] / w2 * 0.5 + 0.5) * w_m1 + 0.5
                vt2_y = (0.5 + 0.5 * V[idx2, 1] / w2) * h_m1 + 0.5

                vt0 = (vt0_x, vt0_y, 0.0)
                vt1 = (vt1_x, vt1_y, 0.0)
                vt2 = (vt2_x, vt2_y, 0.0)

                px_f = px + 0.5
                a, b, g = calc_barycentric(vt0, vt1, vt2, px_f, py_f)

                b0 = a / w0
                b1 = b / w1
                b2 = g / w2
                inv_sum = 1.0 / max(b0 + b1 + b2, 1e-12)
                barycentric_map[py, px, 0] = b0 * inv_sum
                barycentric_map[py, px, 1] = b1 * inv_sum
                barycentric_map[py, px, 2] = b2 * inv_sum

def rasterize_image(V, F, D, width, height, occlusion_truncation=1e-6, use_depth_prior=0):
    device = V.device
    V_np = V.detach().cpu().numpy().astype(np.float32)
    F_np = F.detach().cpu().numpy().astype(np.int32)

    has_depth = False
    D_np = np.zeros((1, 1), dtype=np.float32)
    if use_depth_prior and D is not None and D.numel() > 0:
        has_depth = True
        D_np = D.detach().cpu().numpy().astype(np.float32)

    zbuffer = np.full((height, width), MAX_TOKEN, dtype=np.int64)
    findices = np.zeros((height, width), dtype=np.int32)
    barycentric = np.zeros((height, width, 3), dtype=np.float32)

    _rasterize_triangles(V_np, F_np, width, height, zbuffer, D_np, float(occlusion_truncation), has_depth)
    _compute_barycentrics(V_np, F_np, zbuffer, width, height, findices, barycentric)

    return [torch.from_numpy(findices).to(device), torch.from_numpy(barycentric).to(device)]
