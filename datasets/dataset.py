import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from utils.image_corrupt import image_corrupt
from datasets.robotwin2.robotwin_agilex_dataset import RobotwinAgilexDataset
from datasets.pretrain.egodex_dataset import EgoDexDataset
from datasets.multi_hdf5_vla_dataset import MultiHDF5VLADataset
import h5py
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from datasets.multi_hdf5_vla_dataset import MultiHDF5VLADataset
from datasets.robotwin2.robotwin_agilex_dataset import RobotwinAgilexDataset, HumanToRobotDataset


class VLAConsumerDataset(Dataset):
    """A vision-language-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """

    def __init__(
        self,
        config,
        image_transform,
        num_cameras,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        image_corrupt_severity=None,
        dataset_type=None,
        state_noise_snr=None,
        use_precomp_lang_embed=True,
        upsample_rate=None,
        val=False,
        task_name="adjust_bottle",
        dataset_name="robotwin_agilex",  # Add dataset_name parameter
    ):
        super(VLAConsumerDataset, self).__init__()
        if dataset_type is not None:
            self.dataset_name = dataset_type
        else:
            self.dataset_name = dataset_name

        DATASET_NAMES = {
            "robotwin_agilex", 
            "human_robot_bridge",  # Verify this matches the name in HumanToRobotDataset
            "co_finetune", 
            "egodex"
        }

        # Create the mapping between dataset name and id
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = config["common"]["img_history_size"]
        self.image_transform = image_transform   

        # Initialize dataset based on dataset_name
        if self.dataset_name == "egodex":
            # self.hdf5_dataset = EgoDexDataset(
            #     config=config,
            #     upsample_rate=upsample_rate,
            #     val=val,
            #     use_precomp_lang_embed=use_precomp_lang_embed,
            data_root = os.environ.get("EGODEX_DATA_ROOT")
            if not data_root:
                raise ValueError("EGODEX_DATA_ROOT is missing or empty!")

            self.hdf5_dataset = EgoDexDataset(
                data_root=data_root, 
                config=config,
                upsample_rate=upsample_rate,
                val=val,
                use_precomp_lang_embed=use_precomp_lang_embed,
            )

                # Note: override default paths if needed
                # data_root="/path/to/egodex",
                # stat_path="/path/to/custom/egodex_stat.json",
        elif self.dataset_name == "robotwin_agilex":
            '''
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="multi_task",
                config=config,
                # Note: override default paths
                multi_task_root_dir="/path/to/robotwin2",
            )
            '''
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="single_task",
                task_name="adjust_bottle",
                #added this (by andrew),
                hdf5_folder="aloha-agilex_clean_50/data",
                max_episodes=config.get("dataset", {}).get("max_robot_episodes", 50),
                config=config,
                # Note: override default paths
                single_task_root_dir="/home/ubuntu/RoboTwin/dataset",
            )

        elif self.dataset_name == "human_robot_bridge":
            print("🎯 Loading Vision Pro Human Data (500 episodes)")
            data_path = "/home/ubuntu/human-policy/data/recordings/processed_baseline_robot_frame"
            print(f"📂 Data path: {data_path}/adjust_bottle/.")  # ← Now data_path is defined!
            self.hdf5_dataset = HumanToRobotDataset(
                mode="single_task",
                task_name="adjust_bottle",
                hdf5_folder=".",
                max_episodes=None,
                config=config,
                single_task_root_dir=data_path,
            )


        elif self.dataset_name == "co_finetune":
            print("🧠 Initializing Co-Finetuning: 50% Robot | 50% Human")

            # 1. Robot Dataset (Real AgileX Data)
            # CHANGE 'hdf5_folder' and 'root_dir' to match your actual robot data path
            robot_ds = RobotwinAgilexDataset(
                mode="single_task",
                task_name=task_name,
                hdf5_folder="aloha-agilex_clean_50/data", 
                max_episodes=None,
                config=config,
                single_task_root_dir="/home/ubuntu/RoboTwin/dataset" 
            )

            # 2. Human Dataset (Your New Data)
            # CHANGE 'root_dir' to your generated data path
            human_ds = HumanToRobotDataset(
                mode="single_task",
                task_name=task_name,
                hdf5_folder=".", # Folder inside task_name
                max_episodes=None,
                config=config,
                single_task_root_dir="/home/ubuntu/human-policy/data/recordings/processed_baseline_robot_frame"
            )

            # 3. Mix Them
            self.hdf5_dataset = MultiHDF5VLADataset(
                dataset_list=[robot_ds, human_ds],
                dataset_weights=[0.5, 0.5]  # 50% Robot, 30% Human
            )    
        
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")
            
        print(f"Initialized dataset: {self.dataset_name}")

        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.dataset_type = dataset_type

        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        # self.image_aug_transform = get_image_augmentation()
        self.image_aug = image_aug

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def __len__(self) -> int:
        return len(self.hdf5_dataset)

    def __getitem__(self, index):
        # Get data from backend dataset
        try:
            res = self.hdf5_dataset.get_item(index)
        except Exception as e:
            print(f"Error loading episode {index}: {e}")
            return None
            
        # Add check for res being None, retry a few times if it's None
        retry_count = 0
        max_retries = 5
        while res is None and retry_count < max_retries:
            retry_count += 1
            print(f"Got None data item, retrying {retry_count} time...")
            try:
                res = self.hdf5_dataset.get_item(index)
            except Exception as e:
                print(f"Error during retry data loading: {e}")
                
        # If still None after multiple retries, return a default value to prevent training interruption
        if res is None:
            print(f"Warning: Still unable to get valid data after multiple retries, returning default value")

        data_dict = {}
        data_dict['dataset_name'] = res['dataset_name']
        data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]

        # Process state and action data
        data_dict["states"] = res['states']
        data_dict["actions"] = res['actions']
        data_dict["action_norm"] = res['action_norm']

        # Process images
        if self.dataset_name in ['egodex']:
            # Single camera / stitched image processing
            image_metas = []
            images = res['current_images'][0]
            valid_mask = res.get('current_images_mask', [np.ones(self.img_history_size, dtype=bool)])[0]
            image_metas.append((images, valid_mask))
            
            rearranged_images = []
            for hist_idx in range(self.img_history_size):
                images, valid_mask = image_metas[0]
                if valid_mask[hist_idx]:
                    rearranged_images.append((images[hist_idx], True))
                else:
                    rearranged_images.append((None, False))
        else:
            # Multi-view processing (original logic)
            image_metas = []
            for cam_idx in range(self.num_cameras):
                images = res['current_images'][cam_idx]
                valid_mask = res.get('current_images_mask', np.ones((self.num_cameras, self.img_history_size), dtype=bool))[cam_idx]
                image_metas.append((images, valid_mask))

            rearranged_images = []
            for hist_idx in range(self.img_history_size):
                for cam_idx in range(self.num_cameras):
                    images, valid_mask = image_metas[cam_idx]
                    if valid_mask[hist_idx]:
                        rearranged_images.append((images[hist_idx], True))
                    else:
                        rearranged_images.append((None, False))

        all_pixel_values = []
        for image, valid in rearranged_images:
            image = Image.fromarray(image) if image is not None else None

            if valid and self.auto_adjust_image_brightness:
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)

            # Only apply image augmentation to 50% of the images
            if valid and self.image_aug and (random.random() > 0.5):
                aug_type = random.choice([
                    "corrput_only", "color_only", "both"])
                if aug_type != "corrput_only":
                    image = transforms.ColorJitter(
                        brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                if aug_type != "color_only":
                    image = image_corrupt(image)
                # image = self.image_aug_transform(image)

            pixel_values = self.image_transform(image)
            all_pixel_values.append(pixel_values)

        # Process dino-siglip format images
        pv_example = all_pixel_values[0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in all_pixel_values]
            )
            for k in pv_example
        }
        data_dict["images"] = merged_pixel_values

        if self.use_precomp_lang_embed:
            # All datasets should provide lang_embeds as tensor
            if "lang_embeds" in res:
                data_dict["lang_embeds"] = res["lang_embeds"]
            elif torch.is_tensor(res["instruction"]):
                data_dict["lang_embeds"] = res["instruction"]
            else:
                # Legacy: load from file path
                pt_data = torch.load(res["instruction"])
                data_dict["lang_embeds"] = pt_data["embeddings"].squeeze(0)
                # Load token IDs if available (for reasoning auxiliary loss)
                if "token_ids" in pt_data:
                    data_dict["reasoning_token_ids"] = pt_data["token_ids"]

        # Convert all numpy arrays to torch tensors
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)

        # Verify all data is tensors
        for k, v in data_dict.items():
            assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

        return data_dict

class DataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, use_precomp_lang_embed=True) -> None:
        self.use_precomp_lang_embed = use_precomp_lang_embed
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Initialize batch with common fields
        batch = {
            "states": [],
            "actions": [],
            "action_norm": [],
            "images": [],
            "data_indices": [],
        }
        
        if self.use_precomp_lang_embed:
            lang_embeds = []
            lang_embed_lens = []
        reasoning_token_ids_list = []
        has_reasoning_tokens = False

        # Process each instance in the batch
        for instance in instances:
            # Process numeric data
            keys_to_check = [
                'states', 'actions',
                'action_norm',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            # Process images
            batch["images"].append(instance["images"])
            batch["data_indices"].append(instance["data_idx"])

            if self.use_precomp_lang_embed and "lang_embeds" in instance:
                lang_embeds.append(instance["lang_embeds"])
                lang_embed_lens.append(instance["lang_embeds"].shape[0])
            if "reasoning_token_ids" in instance:
                reasoning_token_ids_list.append(instance["reasoning_token_ids"])
                has_reasoning_tokens = True

        # Stack tensors for numeric data
        keys_to_stack = [
            'states', 'actions',
            'action_norm',
        ]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        # Process dino-siglip format images
        pv_example = batch["images"][0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in batch["images"]]
            )
            for k in pv_example
        }
        batch["images"] = merged_pixel_values

        if self.use_precomp_lang_embed:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(
                lang_embeds,
                batch_first=True,
                padding_value=0)
            input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask
        if has_reasoning_tokens and len(reasoning_token_ids_list) == len(instances):
            batch["reasoning_token_ids"] = torch.nn.utils.rnn.pad_sequence(
                reasoning_token_ids_list,
                batch_first=True,
                padding_value=0)  # pad_token_id=0 for T5

        return batch