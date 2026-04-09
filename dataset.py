import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import mediapipe as mp
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class LateFusionDataset(Dataset):
    """
    Dataset for LateFusionDeepfakeDetector.

    Changes vs previous version:
        - Real augmentation toned down significantly.
          Removed: vflip, RandomPerspective, RandomErasing.
          Reduced: jitter strength, rotation range, noise magnitude.
          Rationale: those transforms destroyed genuine face signal and
          caused the model to be uncertain about real faces at test time
          (real median prob = 0.48 instead of the desired <0.25).

        - Landmark input is now 1405-d (was 1404):
          The final element is a binary "face detected" flag (1.0 or 0.0).
          When MediaPipe fails, coordinates are filled with 0.5 (neutral)
          rather than all zeros, which previously resembled the pattern of
          a real face corner-case and confused the classifier.

        Real augmentation stack (training only):
            - Horizontal flip            (p=0.5)
            - Color jitter               (p=0.6, moderate)
            - Random rotation ±8°        (p=0.4)
            - Gaussian noise             (p=0.3, mild)
            - Random grayscale           (p=0.1)

        Fake augmentation stack (training only, unchanged):
            - Horizontal flip            (p=0.5)
            - Color jitter               (p=0.4, mild)
            - Gaussian noise             (p=0.2)
    """

    def __init__(self, video_paths, labels, is_training=True, num_frames=16):
        self.video_paths = video_paths
        self.labels      = labels
        self.is_training = is_training
        self.num_frames  = num_frames
        self.face_mesh   = None

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if self.face_mesh is None:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )

        video_path = self.video_paths[idx]
        label      = self.labels[idx]
        cap        = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames       = []

        start_frame = max(0, int(total_frames * 0.1))
        end_frame   = min(total_frames - 1, int(total_frames * 0.9))
        if end_frame <= start_frame:
            start_frame, end_frame = 0, total_frames - 1

        indices = np.linspace(start_frame, end_frame, self.num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0 and frame.max() > 0:
                frames.append(frame)

        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((112, 112, 3), dtype=np.uint8)] * self.num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        # ── Spatial tensor ────────────────────────────────────────────────────
        mid_idx       = len(frames) // 2
        mid_bgr       = frames[mid_idx]
        mid_rgb       = cv2.cvtColor(mid_bgr, cv2.COLOR_BGR2RGB)
        spatial_frame = cv2.resize(mid_rgb, (299, 299))

        spatial_tensor = torch.from_numpy(spatial_frame).float() / 255.0
        spatial_tensor = spatial_tensor.permute(2, 0, 1)

        if self.is_training:
            if label == 0:
                # ── Real: moderate augmentation ───────────────────────────────
                # NOTE: vflip, RandomPerspective, RandomErasing removed because
                # they destroyed face geometry and pushed real probs toward 0.5.
                if random.random() < 0.5:
                    spatial_tensor = TF.hflip(spatial_tensor)

                if random.random() < 0.6:
                    jitter = T.ColorJitter(
                        brightness=0.25, contrast=0.25,
                        saturation=0.15, hue=0.06
                    )
                    spatial_tensor = jitter(spatial_tensor)

                if random.random() < 0.4:
                    angle = random.uniform(-8, 8)
                    spatial_tensor = TF.rotate(spatial_tensor, angle)

                if random.random() < 0.3:
                    noise = torch.randn_like(spatial_tensor) * 0.02
                    spatial_tensor = torch.clamp(spatial_tensor + noise, 0.0, 1.0)

                if random.random() < 0.1:
                    spatial_tensor = TF.rgb_to_grayscale(
                        spatial_tensor, num_output_channels=3
                    )

            else:
                # ── Fake: mild augmentation (unchanged) ───────────────────────
                if random.random() < 0.5:
                    spatial_tensor = TF.hflip(spatial_tensor)

                if random.random() < 0.4:
                    jitter = T.ColorJitter(
                        brightness=0.2, contrast=0.2,
                        saturation=0.1, hue=0.05
                    )
                    spatial_tensor = jitter(spatial_tensor)

                if random.random() < 0.2:
                    noise = torch.randn_like(spatial_tensor) * 0.02
                    spatial_tensor = torch.clamp(spatial_tensor + noise, 0.0, 1.0)

        spatial_tensor = TF.normalize(spatial_tensor,
                                      mean=[0.5, 0.5, 0.5],
                                      std =[0.5, 0.5, 0.5])

        # ── Temporal tensor ───────────────────────────────────────────────────
        temporal_frames = []
        for frame_bgr in frames:
            frame_rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (224, 224))
            temporal_frames.append(resized_frame)

        temporal_stack  = np.stack(temporal_frames, axis=0)
        temporal_tensor = torch.from_numpy(temporal_stack).float() / 255.0
        temporal_tensor = temporal_tensor.permute(3, 0, 1, 2)

        mean = torch.tensor([0.43216, 0.394666, 0.37645 ]).view(3, 1, 1, 1)
        std  = torch.tensor([0.22803, 0.22145,  0.216989]).view(3, 1, 1, 1)
        temporal_tensor = (temporal_tensor - mean) / std

        # ── Landmark tensor (1405-d: 468×3 coords + 1 detection flag) ─────────
        # When face is not detected, coordinates are set to 0.5 (neutral
        # midpoint) rather than 0.0 (which resembled a specific real pattern
        # and confused the classifier). The final flag element lets the
        # LandmarkExtractor learn to down-weight no-detection samples.
        results = self.face_mesh.process(mid_rgb)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmark_array = np.array(landmarks, dtype=np.float32)
            # Use np.float32 for the flag so np.append doesn't promote to float64
            landmark_array = np.append(landmark_array, np.float32(1.0))
        else:
            landmark_array = np.full(468 * 3, 0.5, dtype=np.float32)
            landmark_array = np.append(landmark_array, np.float32(0.0))

        # .float() is a safety net in case any upstream path still returns float64
        landmark_tensor = torch.from_numpy(landmark_array).float()
        label_tensor    = torch.tensor(label, dtype=torch.long)

        return {
            'spatial' : spatial_tensor,
            'temporal': temporal_tensor,
            'landmark': landmark_tensor,
            'label'   : label_tensor
        }

    def __del__(self):
        if getattr(self, 'face_mesh', None) is not None:
            self.face_mesh.close()