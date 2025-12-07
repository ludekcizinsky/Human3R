import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
import json
import subprocess
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import roma
from tqdm import tqdm

# Set random seed for reproducibility.
random.seed(42)

class _ProgressList(list):
    """List wrapper that updates a tqdm bar each iteration."""

    def __init__(self, data, progress_bar):
        super().__init__(data)
        self._progress_bar = progress_bar

    def __iter__(self):
        for item in super().__iter__():
            self._progress_bar.update(1)
            yield item

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to use. Default is None (use all images).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for input images. Default is 1 (use all images).",
    )
    parser.add_argument(
        "--reset_interval", 
        type=int, 
        default=10000000
        )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False
    )
    return parser.parse_args()


def prepare_input(
    img_paths, 
    img_mask, 
    size, 
    raymaps=None, 
    raymap_mask=None, 
    revisit=1, 
    update=True, 
    img_res=None, 
    reset_interval=100
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu") # if use pseudo K

    views = []
    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views

def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname

def _to_numpy(tensor_like):
    if tensor_like is None:
        return None
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like.detach().cpu().numpy()
    return tensor_like


def save_output(outputs, outdir, revisit=1, subsample=1):

    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod, get_camera_parameters
    from src.dust3r.utils import SMPL_Layer, render_meshes
    from src.dust3r.utils.image import unpad_image
    from viser_utils import get_color

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]
    frame_indices = []
    for view in outputs["views"]:
        idx = view["idx"]
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        frame_indices.append(int(idx))

    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    # reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # keeps only reset_mask=False pr_poses
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))


    # Save camera parameters
    # - Extrinsics
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)
    R_w2c = R_c2w.permute(0, 2, 1)
    t_w2c = -torch.einsum("bij,bj->bi", R_w2c, t_c2w)

    # - Intrinsics: estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # - save to disk
    cameras_to_save = {
        "frame_idx": np.asarray(frame_indices, dtype=np.int32),
        "R_world2cam": _to_numpy(R_w2c),
        "t_world2cam": _to_numpy(t_w2c),
        "K": _to_numpy(intrinsics_tosave),
    }

    cameras_path = os.path.join(outdir, "cameras.npz")
    np.savez(cameras_path, **cameras_to_save)


    # Parse rgb frames from outputs
    rgb_frames = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]


    # Get SMPL parameters from outputs
    smpl_shape = [output.get(
        "smpl_shape", torch.empty(1,0,10))[0] for output in outputs["pred"]]
    smpl_rotvec = [roma.rotmat_to_rotvec(
        output.get(
            "smpl_rotmat", torch.empty(1,0,53,3,3))[0]) for output in outputs["pred"]]
    smpl_transl = [output.get(
        "smpl_transl", torch.empty(1,0,3))[0] for output in outputs["pred"]]
    smpl_expression = [output.get(
        "smpl_expression", [None])[0] for output in outputs["pred"]]
    smpl_id = [output.get(
        "smpl_id", torch.empty(1,0))[0] for output in outputs["pred"]]

        
    # SMPL layer
    smpl_layer = SMPL_Layer(type='smplx', 
                            gender='neutral', 
                            num_betas=smpl_shape[0].shape[-1], 
                            kid=False, 
                            person_center='head')
    smpl_faces = smpl_layer.bm_x.faces


    # Save the per person and per frame smplx parameters
    per_person_state = {}
    for f_id in range(B):

        # Parse number of humans in the given frame
        n_humans_i = smpl_shape[f_id].shape[0]

        # TODO: need to figure out what this exactly does
        with torch.no_grad():
            smpl_out = smpl_layer(
                smpl_rotvec[f_id], 
                smpl_shape[f_id], 
                smpl_transl[f_id], 
                K=intrinsics_tosave[f_id].expand(n_humans_i, -1 , -1), 
                expression=smpl_expression[f_id]
            )

        # Visualisation inputs
        # - Get the rgb image
        color = rgb_frames[f_id].numpy()
        img_array_np = (color * 255).astype(np.uint8)

        # - Get the intrinsic parameters for the current frame 
        intrins = intrinsics_tosave[f_id].numpy()
        intrins_dict = {
            'focal': intrins[[0, 1], [0, 1]],
            'princpt': intrins[[0, 1], [-1, -1]],
        }

        pr_verts = [t.numpy() for t in smpl_out['smpl_v3d'].unbind(0)]

        # To be saved for later purposes
        # - Get the rotation vectors and pelvis translation in camera coordinates 
        rotvec_cam = smpl_rotvec[f_id].detach().cpu().clone()
        transl_pelvis_cam = smpl_out["smpl_transl_pelvis"].detach().cpu()  # [n_h,1,3]

        # Parse person IDs
        raw_person_ids = smpl_id[f_id]
        if raw_person_ids is None:
            person_ids = torch.arange(n_humans_i, dtype=torch.int32)
        else:
            person_ids = raw_person_ids if isinstance(
                raw_person_ids, torch.Tensor
            ) else torch.tensor(raw_person_ids)
            if isinstance(person_ids, torch.Tensor):
                if person_ids.numel() == 0:
                    person_ids = torch.arange(n_humans_i, dtype=torch.int32)
                else:
                    person_ids = person_ids.detach().cpu().to(torch.int32)


        for h_idx in range(n_humans_i):
            pid = int(person_ids[h_idx].item())

            # Create directories for the current person if they don't exist
            if pid not in per_person_state:
                person_root = os.path.join(outdir, f"{pid:02d}")
                param_dir = os.path.join(person_root, "smplx_params")
                frame_dir = os.path.join(person_root, "frames")
                os.makedirs(param_dir, exist_ok=True)
                os.makedirs(frame_dir, exist_ok=True)
                per_person_state[pid] = {
                    "param_dir": param_dir,
                    "frame_dir": frame_dir,
                    "frame_counter": 0,
                }
            state = per_person_state[pid]

            # Save the SMPLX parameters
            # - Reorder to SMPL-X standard ordering (root, body, jaw, eyes, hands).
            rot_reordered = torch.zeros((55, 3), dtype=torch.float32)
            rot_reordered[0] = rotvec_cam[h_idx, 0]
            rot_reordered[1:22] = rotvec_cam[h_idx, 1:22]
            rot_reordered[22] = rotvec_cam[h_idx, 52]  # jaw
            rot_reordered[25:40] = rotvec_cam[h_idx, 22:37]  # left hand
            rot_reordered[40:55] = rotvec_cam[h_idx, 37:52]  # right hand

            # - Save to disk
            #   Store both the raw translation given to SMPL Layer (camera frame, primary keypoint)
            #   and the pelvis translation after the layer's internal offset for reference.
            trans_cam = smpl_out["smpl_transl"][h_idx].detach().cpu().float().numpy().tolist()
            trans_pelvis_cam_single = transl_pelvis_cam[h_idx, 0].detach().cpu().float().numpy().tolist()
            params = {
                "betas": smpl_shape[f_id][h_idx].detach().cpu().numpy().tolist(),
                "root_pose": rot_reordered[0].numpy().tolist(),
                "body_pose": rot_reordered[1:22].numpy().tolist(),
                "jaw_pose": rot_reordered[22].numpy().tolist(),
                "leye_pose": [0.0, 0.0, 0.0],
                "reye_pose": [0.0, 0.0, 0.0],
                "lhand_pose": rot_reordered[25:40].numpy().tolist(),
                "rhand_pose": rot_reordered[40:55].numpy().tolist(),
                # translation expected by vanilla SMPL-X layer (camera frame, pelvis/root)
                "trans": trans_cam,
                # pelvis translation as produced by the custom SMPL_Layer (kept for reference/compat)
                "trans_pelvis": trans_pelvis_cam_single,
            }
            json_path = os.path.join(state["param_dir"], f"{frame_indices[f_id]:05d}.json")
            with open(json_path, "w") as f:
                json.dump(params, f)

            # Render the SMPL mesh on the image for visualization
            smpl_rend_single = render_meshes(
                img_array_np.copy(),
                [pr_verts[h_idx]],
                [smpl_faces],
                intrins_dict,
                color=[get_color(pid) / 255],
            )
            side_by_side = np.concatenate(
                [
                    img_array_np,
                    smpl_rend_single,
                ],
                1,
            )

            # Update the frame counter for the current person
            frame_seq_idx = state["frame_counter"]
            state["frame_counter"] += 1

            # Save the side-by-side visualization image
            frame_path = os.path.join(state["frame_dir"], f"{frame_seq_idx:06d}.png")
            iio.imwrite(frame_path, side_by_side)

    # Create videos from the saved visualization frames
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    output_fps = max(1, 20 // subsample)
    for pid, state in per_person_state.items():
        if state["frame_counter"] == 0:
            continue
        frames_dir = state["frame_dir"]
        video_path = os.path.join(os.path.dirname(frames_dir), "pose_visualized.mp4")
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(output_fps),
            "-i",
            f"{frames_dir}/%06d.png",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vcodec",
            "h264",
            "-preset",
            "fast",
            "-profile:v",
            "baseline",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-b:v",
            "5000k",
            video_path,
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        shutil.rmtree(frames_dir, ignore_errors=True)
    


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return
    
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Prepare input views.
    print("Preparing input views...")
    img_res = getattr(model, 'mhmr_img_res', None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval
    )

    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Run inference.
    print("Running inference...")
    start_time = time.time()

    progress_bar = tqdm(total=len(views), desc="Inference", unit="frame")
    try:
        outputs, _ = inference_recurrent_lighter(
            _ProgressList(views, progress_bar), model, device, use_ttt3r=args.use_ttt3r
        )
    finally:
        progress_bar.close()
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    save_output(outputs, args.output_dir, 1, args.subsample)

def main():
    args = parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()
