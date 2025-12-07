import os
import sys
import glob
import argparse
from collections import defaultdict
from pathlib import Path
from PIL import Image
import json
from typing import List, Tuple
import numpy as np
import torch
import pyrender
import trimesh

# insert new path to sys
sys.path.insert(0, "/home/cizinsky/LHM")
from LHM.models.rendering.smpl_x_voxel_dense_sampling import SMPLXVoxelMeshModel

def load_human3r_cameras(cameras_npz_path):
    """
    Load Human3R camera parameters (cameras.npz) and return torch tensors.

    Returns:
        c2ws: [N, 4, 4] camera-to-world transforms.
        intrs: [N, 4, 4] intrinsics with bottom row [0,0,0,1].
    """
    data = np.load(cameras_npz_path)
    required_keys = ["R_world2cam", "t_world2cam", "K"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {cameras_npz_path}")

    order = None
    if "frame_idx" in data:
        order = np.argsort(data["frame_idx"])

    def _maybe_reorder(arr):
        return arr if order is None else arr[order]

    R_w2c = torch.as_tensor(_maybe_reorder(data["R_world2cam"]), dtype=torch.float32)
    t_w2c = torch.as_tensor(_maybe_reorder(data["t_world2cam"]), dtype=torch.float32)
    K = torch.as_tensor(_maybe_reorder(data["K"]), dtype=torch.float32)

    R_c2w = R_w2c.transpose(1, 2)
    t_c2w = -torch.einsum("bij,bj->bi", R_c2w, t_w2c)

    N = R_c2w.shape[0]
    c2ws = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
    c2ws[:, :3, :3] = R_c2w
    c2ws[:, :3, 3] = t_c2w

    intrs = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
    intrs[:, :3, :3] = K

    return c2ws, intrs

def prepare_motion_seqs_human3r(
    motion_seqs_dir : Path,
    bg_color=1.0,
    motion_size=3000,  # only support 100s videos
):

    # motion_seqs_dir: directory of smplx_params predicted by human3r
    motion_seqs = sorted(glob.glob(os.path.join(motion_seqs_dir, "*.json")))
    motion_seqs = motion_seqs[:motion_size]

    # load smplx params
    smplx_params, bg_colors = [], []

    for idx, smplx_path in enumerate(motion_seqs):
        with open(smplx_path) as f:
            smplx_raw_data = json.load(f)
            smplx_param = {
                k: torch.FloatTensor(v)
                for k, v in smplx_raw_data.items()
                if "pad_ratio" not in k
            }

        smplx_param["expr"] = torch.FloatTensor([0.0] * 100)
        if "trans_offset" not in smplx_param:
            smplx_param["trans_offset"] = torch.zeros_like(smplx_param["trans"])

        bg_colors.append(bg_color)
        smplx_params.append(smplx_param)

    cameras_npz_path = motion_seqs_dir.parent.parent / "cameras.npz"
    c2ws, intrs = load_human3r_cameras(cameras_npz_path)
    bg_colors = (
        torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    )  # [N, 3]

    smplx_params_tmp = defaultdict(list)
    for smplx in smplx_params:
        for k, v in smplx.items():
            smplx_params_tmp[k].append(v)
    for k, v in smplx_params_tmp.items():
        smplx_params_tmp[k] = torch.stack(v)  # [Nv, xx, xx]
    # Betas are frame-invariant; keep first frame only so shape is [Nv=1, beta_dim]
    if "betas" in smplx_params_tmp:
        # Frame-invariant shape params: squeeze frame dimension so betas shape is [beta_dim]
        smplx_params_tmp["betas"] = smplx_params_tmp["betas"][:1].squeeze(0)
    smplx_params = smplx_params_tmp

    # add batch dim
    for k, v in smplx_params.items():
        smplx_params[k] = v.unsqueeze(0)
    c2ws = c2ws.unsqueeze(0)
    intrs = intrs.unsqueeze(0)
    bg_colors = bg_colors.unsqueeze(0)

    motion_seqs_ret = {}
    motion_seqs_ret["render_c2ws"] = c2ws
    motion_seqs_ret["render_intrs"] = intrs
    motion_seqs_ret["render_bg_colors"] = bg_colors
    motion_seqs_ret["smplx_params"] = smplx_params
    motion_seqs_ret["motion_seqs"] = motion_seqs

    return motion_seqs_ret


def smplx_base_vertices_in_camera(
    smplx_model,
    smplx_params: dict,
    pid: int,
    frame_idx: int,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Compute base (non-upsampled) SMPL-X vertices in camera coordinates for a given person and frame.
    Assumes the provided SMPL-X parameters are already expressed in the camera frame.
    Returns: [V_base, 3] or None on failure.
    """
    try:
        def _pad_or_truncate(t: torch.Tensor, target_dim: int | None, label: str) -> torch.Tensor:
            if target_dim is None:
                return t
            cur = t.shape[-1]
            if cur == target_dim:
                return t
            if cur > target_dim:
                print(f"[DEBUG] Truncating {label} from {cur} to {target_dim}")
                return t[..., :target_dim]
            pad = torch.zeros(*t.shape[:-1], target_dim - cur, device=t.device, dtype=t.dtype)
            return torch.cat([t, pad], dim=-1)

        layer = getattr(smplx_model, "smplx_layer", None)
        if layer is None and hasattr(smplx_model, "layer"):
            layer = smplx_model.layer.get("neutral", None)
        if layer is None:
            raise AttributeError("SMPLX model has no base smplx_layer")
        layer = layer.to(device)

        expected_beta_dim = getattr(layer, "num_betas", None)
        if expected_beta_dim is None and hasattr(layer, "shapedirs"):
            try:
                expected_beta_dim = int(layer.shapedirs.shape[-1])
            except Exception:
                expected_beta_dim = None
        expected_expr_dim = getattr(layer, "num_expression_coeffs", None)
        if expected_expr_dim is None and hasattr(layer, "expr_dirs"):
            try:
                expected_expr_dim = int(layer.expr_dirs.shape[-1])
            except Exception:
                expected_expr_dim = None

        # We want the head joint to land at the raw translation predicted by the
        # inference pipeline. With vanilla SMPL-X the global rotation is around
        # the pelvis, so we solve for a pelvis translation that places the head
        # at the desired camera coordinate.
        transl_saved = smplx_params["trans"][pid : pid + 1, frame_idx]  # pelvis-based translation
        trans_raw_all = smplx_params.get("trans_raw", None)
        target_head_cam = (
            trans_raw_all[pid : pid + 1, frame_idx] if trans_raw_all is not None else transl_saved
        )

        base_params = {
            "global_orient": smplx_params["root_pose"][pid : pid + 1, frame_idx],
            "body_pose": smplx_params["body_pose"][pid : pid + 1, frame_idx],
            "jaw_pose": smplx_params["jaw_pose"][pid : pid + 1, frame_idx],
            "leye_pose": smplx_params["leye_pose"][pid : pid + 1, frame_idx],
            "reye_pose": smplx_params["reye_pose"][pid : pid + 1, frame_idx],
            "left_hand_pose": smplx_params["lhand_pose"][pid : pid + 1, frame_idx],
            "right_hand_pose": smplx_params["rhand_pose"][pid : pid + 1, frame_idx],
            "betas": _pad_or_truncate(smplx_params["betas"][pid : pid + 1], expected_beta_dim, "betas"),
        }
        if "expr" in smplx_params:
            expr = smplx_params["expr"][pid : pid + 1, frame_idx]
            base_params["expression"] = _pad_or_truncate(expr, expected_expr_dim, "expr")

        # First pass with zero translation to measure head location after global_orient.
        zero_params = dict(base_params)
        zero_params["transl"] = torch.zeros_like(transl_saved)
        zero_out = layer(**{k: v.to(device) for k, v in zero_params.items()})
        head_idx = 15  # matches SMPL_Layer person_center="head"
        head_offset = zero_out.joints[:, head_idx]  # [1,3] in camera frame with zero transl
        target_head_cam = target_head_cam.to(head_offset.device)

        # Solve for pelvis translation that sends head to target_head_cam.
        solved_transl = target_head_cam - head_offset

        solve_params = dict(base_params)
        solve_params["transl"] = solved_transl
        output = layer(**{k: v.to(device) for k, v in solve_params.items()})
        return output.vertices[0]
    except Exception as e:
        print(f"[DEBUG] Could not compute base SMPL-X verts in camera: {e}")
        return None

def overlay_smplx_mesh_pyrender(
    images: torch.Tensor,
    smplx_params: dict,
    smplx_model,
    intr: torch.Tensor,
    device: torch.device,
    mesh_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    mesh_alpha: float = 0.7,
) -> torch.Tensor:
    """
    Render SMPL-X meshes with trimesh+pyrender and alpha-blend them over images.
    
    images: [F, H, W, 3] float in [0,1]
    smplx_params: dict with shapes [P, F, ...]
    intr: [3,3] or [4,4] intrinsics
    mesh_color: RGB in [0,1]; mesh_alpha: opacity for the mesh layer
    """

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    layer = getattr(smplx_model, "smplx_layer", None)
    if layer is None and hasattr(smplx_model, "layer"):
        layer = smplx_model.layer.get("neutral", None)
    faces = getattr(layer, "faces", None) if layer is not None else None
    if faces is None:
        print("[WARN] SMPL-X faces not found, skipping mesh overlay.")
        return images
    faces_np = np.asarray(faces, dtype=np.int64)

    intr_cpu = intr.detach().cpu()
    if intr_cpu.shape[-2:] == (4, 4):
        intr_cpu = intr_cpu[:3, :3]
    fx, fy, cx, cy = (
        float(intr_cpu[0, 0]),
        float(intr_cpu[1, 1]),
        float(intr_cpu[0, 2]),
        float(intr_cpu[1, 2]),
    )

    num_frames = images.shape[0]
    num_people = smplx_params["betas"].shape[0]
    H, W = images.shape[1], images.shape[2]

    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    except Exception as e:
        print(f"[WARN] Could not initialise pyrender renderer: {e}")
        return images

    out_frames: List[torch.Tensor] = []
    out_masks: List[torch.Tensor] = []
    try:
        for fi in range(num_frames):
            base_img = (images[fi].detach().cpu().numpy() * 255).astype(np.uint8)
            depth_map = np.ones((H, W)) * np.inf
            overlay_img = base_img.astype(np.float32)
            frame_mask = torch.zeros((H, W, 1), device=images.device, dtype=images.dtype)

            for pid in range(num_people):
                cam_verts = smplx_base_vertices_in_camera(
                    smplx_model, smplx_params, pid, fi, device
                )
                if cam_verts is None:
                    continue
                verts_np = cam_verts.detach().cpu().numpy()

                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode="BLEND",
                    baseColorFactor=[
                        float(mesh_color[0]),
                        float(mesh_color[1]),
                        float(mesh_color[2]),
                        float(mesh_alpha),
                    ],
                )
                mesh = trimesh.Trimesh(verts_np, faces_np, process=False)
                rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                mesh.apply_transform(rot)
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

                scene = pyrender.Scene(
                    bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.5, 0.5, 0.5)
                )
                scene.add(mesh, "mesh")
                camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
                scene.add(camera, pose=np.eye(4))
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
                scene.add(light, pose=np.eye(4))

                color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                valid_mask = (rend_depth < depth_map) & (rend_depth > 0)
                depth_map[valid_mask] = rend_depth[valid_mask]
                valid_mask = valid_mask[..., None]
                overlay_img = valid_mask * color[..., :3] + (1.0 - valid_mask) * overlay_img
                frame_mask = torch.logical_or(frame_mask.bool(), torch.from_numpy(valid_mask).to(device=images.device)).to(images.dtype)

            overlay_tensor = (
                torch.from_numpy(overlay_img).to(device=images.device, dtype=images.dtype) / 255.0
            )
            out_frames.append(overlay_tensor)
            out_masks.append(frame_mask)
    finally:
        renderer.delete()

    return torch.stack(out_frames, dim=0), torch.stack(out_masks, dim=0)

def load_img(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    return arr.to(device)


def load_mask(path: Path, device: torch.device, eps: float = 0.05) -> torch.Tensor:
    arr = torch.from_numpy(np.array(Image.open(path))).float()  # HxWxC or HxW
    if arr.dim() == 2:
        arr = arr.unsqueeze(-1) / 255.0  # HxWx1
        return arr.to(device) # already binary mask

    if arr.shape[-1] == 4:
        arr = arr[..., :3] # drop alpha
    # Foreground is any pixel whose max channel exceeds eps*255
    mask = (arr.max(dim=-1).values > eps * 255).float()  # HxW
    return mask.to(device).unsqueeze(-1)  # HxWx1, range [0,1]


if __name__ == "__main__":

    smplx_model = SMPLXVoxelMeshModel(
        "/scratch/izar/cizinsky/pretrained/pretrained_models/human_model_files",
        gender="neutral",
        subdivide_num=1,
        shape_param_dim=10,
        expr_param_dim=100,
        cano_pose_type=1,
        dense_sample_points=40000,
        apply_pose_blendshape=False,
    )
    device = "cuda"

    # load frames
    frames_dir_path = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair19_piggyback/lhm/frames"
    frames_dir = Path(frames_dir_path)
    frame_paths = sorted(frames_dir.glob("*.png"))[:1]
    frames = [load_img(fp, device) for fp in frame_paths]
    frames_tensor = torch.stack(frames, dim=0)
    print(f"Shape of frames tensor: {frames_tensor.shape} aned min {frames_tensor.min()} and max {frames_tensor.max()}")

    # load motion sequences
    motion_dir_path = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair19_piggyback/lhm/motion_human3r"
    motion_dir = Path(motion_dir_path)
    tids = [p.name for p in motion_dir.iterdir() if p.is_dir()]
    motion_params_per_person = list()
    for tid in tids:
        smplx_tid_dir = motion_dir / tid / "smplx_params"
        print(f"reading {smplx_tid_dir}")
        motion_seq_tid = prepare_motion_seqs_human3r(smplx_tid_dir)
        motion_params_per_person.append(motion_seq_tid)
        
        print(f"Loaded motion sequence for {tid} and it has the following set of keys: {list(motion_seq_tid.keys())}")

        
    # parse intrinsics
    intrinsics = motion_params_per_person[0]["render_intrs"][0, 0]
    print(f"Intrinsics shape: {intrinsics.shape}")

    # parse smplx params
    smplx_params_keys = motion_params_per_person[0]["smplx_params"].keys()
    joined_smplx_params = dict()
    for k in smplx_params_keys:
        all_smplx_params = torch.cat([m["smplx_params"][k] for m in motion_params_per_person], dim=0)
        joined_smplx_params[k] = all_smplx_params

    firstx_smplx = {}
    for k, v in joined_smplx_params.items():
        if k == "betas":
            firstx_smplx[k] = v  # [P, beta_dim]
        else:
            firstx_smplx[k] = v[:, :1]  # [P, 1, ...]

    # Initial render
    overlay_init, masks_init = overlay_smplx_mesh_pyrender(frames_tensor, firstx_smplx, smplx_model, intrinsics, device)
    rendered_mask = masks_init[0]

    # Load corresponding ground truth mask
    mask_dir = Path("/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19/seg/img_seg_mask/4/all")
    mask_paths = sorted(mask_dir.glob("*.png"))[:1]
    masks = [load_mask(mp, device) for mp in mask_paths]
    gt_mask = torch.stack(masks, dim=0)[0]

    # Compute IoU
    intersection = (rendered_mask * gt_mask).sum()
    union = ((rendered_mask + gt_mask) > 0).float().sum()
    iou = intersection / union
    print(f"IoU: {iou.item():.4f}")

    # Save to disk for visual comparison
    out_path = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair19_piggyback/lhm/debug/dev"
    out_file = Path(out_path) / f"rendered_smplx_mask.png"
    overlay = rendered_mask*0.5 + gt_mask*0.5
    img_columns = [rendered_mask, gt_mask, overlay]
    img_grid = torch.cat(img_columns, dim=1) # concatenate along width, shape [H, W*3, 1]
    img_grid_np = (img_grid.squeeze(-1).detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_grid_np)
    img.save(out_file)