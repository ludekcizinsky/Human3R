# Modified from Multi-HMR
# Preprocess 3DPW dataset for human evaluation, and saves the annotations
# files (i.e., 3dpw_test.pkl) in [ANNOT_DIR].
# Usage: python -m eval.dataset.prepare_3dpw "create_annots()"

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import pickle
import torch
import smplx
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image, ImageFile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))
from dust3r.smpl_model import SMPLX_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid "OSError: image file is truncated"
from torch.utils.data import Dataset
import roma

ANNOT_DIR = "eval/global_human/annots/"
THREEDPW_DIR = "/path/to/3DPW"

class THREEDPW(Dataset):
    def __init__(self,
                 split='test',
                 training=False,
                 img_size=512,
                 root_dir=THREEDPW_DIR,
                 force_build_dataset=0,
                 subsample=1,
                 *args, **kwargs
                 ):
        super().__init__()
        
        self.name = '3dpw'
        self.annotations_dir = ANNOT_DIR
        self.training = training
        self.img_size = img_size
        self.subsample = subsample

        assert split in ['test']
        
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(self.root_dir, 'imageFiles')
        self.annot_file = os.path.join(self.annotations_dir, f"{self.name}_{split}.pkl")
        self.force_build_dataset = force_build_dataset

        self.annots = None
        if self.force_build_dataset or not os.path.isfile(self.annot_file):
            self.annots = self.build_dataset()
        if self.annots is None:
            with open(self.annot_file, 'rb') as f:
                self.annots = pickle.load(f)

        self.imagenames = list(self.annots.keys())
        self.imagenames.sort()

        if self.subsample > 1:
            self.imagenames = [self.imagenames[k] for k in np.arange(0,len(self.imagenames),self.subsample).tolist()]

    @torch.no_grad()
    def build_dataset(self):
        print(f"Bulding annotation file for: {self.name} - {self.split}")

        imagename2annot = {}

        smpl_layer_male = smplx.create(SMPLX_DIR, 'smpl', gender='male')
        smpl_layer_female = smplx.create(SMPLX_DIR, 'smpl', gender='female')

        # Filenames
        fns = os.listdir(os.path.join(self.root_dir, 'sequenceFiles', self.split))
        fns.sort()

        error = 0
        max_human = 0
        # Loop across sequence
        for fn in tqdm(fns):
            # Metadata
            with open(os.path.join(self.root_dir, 'sequenceFiles', self.split, fn), 'rb') as f:
                metadata = pickle.load(f, encoding='latin1')

            # Camera intrinsics
            K = metadata['cam_intrinsics']
            focal = np.asarray([K[0,0],K[1,1]])
            princpt = np.asarray([K[0,-1],K[1,-1]])

            # Loop across time
            seq_len = len(metadata['poses'][0])
            n_person = len(metadata['genders'])
            for k in range(seq_len):
                # Image
                seq_name = fn.replace('.pkl', '')
                img_path = os.path.join(seq_name, f"image_{k:05d}.jpg")

                # Resolution
                width, height = Image.open(os.path.join(self.image_dir, img_path)).size

                # Camera extrinsics
                T = metadata['cam_poses'][k]
                R, t = T[:3,:3], T[:3,-1]
                
                # Loop across person
                persons = []
                for i in range(n_person):
                    # gt
                    valid = metadata['campose_valid'][i][k]
                    if valid == 0:
                        continue
                    poses = metadata['poses'][i][k].reshape(24,3)
                    trans = metadata['trans'][i][k]
                    shape = metadata['betas'][i][:10]
                    gender = metadata['genders'][i]
                    poses2d = metadata['poses2d'][i].transpose(0, 2, 1)[k] # [18,3] - openpose
                    idx_valid2d = np.where(poses2d[:,-1] > 0.5)[0]
                    poses2d = poses2d[idx_valid2d,:2]
                    gender_ = 'male' if gender == 'm' else 'female'

                    # apply camera extrinsic (rotation)
                    body_pose = poses[1:]
                    root_pose = poses[0]
                    root_pose = roma.rotvec_to_rotmat(torch.Tensor(root_pose)).numpy()
                    root_pose = R @ root_pose
                    root_pose = roma.rotmat_to_rotvec(torch.Tensor(root_pose)).numpy()

                    # get mesh w/0 transl
                    smpl_layer_ = smpl_layer_male if gender_ == 'male' else smpl_layer_female
                    out = smpl_layer_(global_orient=torch.from_numpy(root_pose).reshape(1,-1).float(),
                                      body_pose=torch.from_numpy(body_pose).reshape(1,-1).float(),
                                      betas=torch.from_numpy(shape).reshape(1,-1).float()
                                    )
                    # apply trans
                    v3d, j3d = out.vertices.numpy().reshape(-1,3), out.joints.numpy().reshape(-1,3)
                    mesh_cam, joint_cam = v3d + trans.reshape(1,3), j3d + trans.reshape(1,3)

                    # apply camera exrinsic (translation) - it will compenstate rotation (translation from origin to root joint was not canceled)
                    root_cam = joint_cam[0,None,:]
                    mesh_cam = mesh_cam - root_cam + np.dot(R, root_cam.transpose(1,0)).transpose(1,0) + t # camera-centered coordinate system

                    # find real transl in camera coordinate system
                    trans = (mesh_cam - v3d)[0]

                    # Append
                    person = {
                        # SMPL pseudo-GT
                        'smpl_root_pose': root_pose.reshape(1,3).astype(np.float32), # axis-angle
                        'smpl_body_pose': body_pose.reshape(23,3).astype(np.float32), # axis-angle
                        'smpl_shape': shape.reshape(10).astype(np.float32),
                        'smpl_transl': trans.reshape(3).astype(np.float32),
                        'smpl_gender': gender_,
                    }
                    persons.append(person)

                if len(persons) > max_human:
                    max_human = len(persons)     

                # Append
                if len(persons) > 0:
                    imagename2annot[img_path] = {
                            # Camera
                            'focal': focal.astype(np.float32).reshape(2),
                            'princpt': princpt.astype(np.float32).reshape(2),
                            'size': np.asarray([width, height]).astype(np.int32).reshape(2),
                            # Humans
                            'humans': persons,
                            'cam_poses': T.astype(np.float32),
                        }
                    
        print(f"max_human: {max_human}")
    
        # Saving
        os.makedirs(os.path.dirname(self.annot_file), exist_ok=True)
        print(f"Saving into {self.annot_file}")
        with open(self.annot_file, 'wb') as f:
            pickle.dump(imagename2annot, f, protocol=pickle.HIGHEST_PROTOCOL)

        return imagename2annot

def create_annots():
    dataset = THREEDPW(split='test', force_build_dataset=1)

if __name__ == "__main__":
    exec(sys.argv[1])