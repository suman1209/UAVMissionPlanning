# standard imports
import math
import os

import yaml
import airsim
import numpy as np
import sys
import pathlib
from PIL import Image, ImageDraw, ImageFont
import io
import datetime
import gym
from gym import spaces
from typing import Union, List
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# local imports
from utils.gym_utils import Pose, Observation
from simulator_interface.unreal_process_manager import UnrealProcessManager
from utils.utils import get_unreal_binary, get_active_settings_file 
import os
import yaml
import pathlib
import numpy as np

class AirSimClientWrapper(gym.Env):
    def __init__(self, unreal_binary_path, startup_wait_sec=15, rendering_enabled=False, settings_file_path=None, topdown_map_altitude=200):
        super().__init__()
        self.unreal_manager = UnrealProcessManager(
            unreal_binary_path=unreal_binary_path,
            startup_wait_sec=startup_wait_sec,
            settings_file_path=settings_file_path
        )
        self.sim_mode = self.unreal_manager.sim_mode
        if self.sim_mode == "Multirotor":
            self.client = airsim.MultirotorClient()
        else:
            self.client = airsim.VehicleClient()
        self.connected = False
        self.rendering_enabled = rendering_enabled
        self.map_image = None
        # Define gym environment spaces
        self.action_space = spaces.Box(
            low=np.array([-100, -100, -100, -np.pi]),  # x, y, z, yaw limits
            high=np.array([100, 100, 0, np.pi]),       # Note: z=0 is ground level, yaw in radians
            dtype=np.float32
        )

        # Observation space will be defined after connecting
        self.observation_space = None
        self.voxel_grid = None
        self.topdown_map_altitude = topdown_map_altitude

    def connect(self):
        self.unreal_manager.reset()
        self.unreal_manager.start_process(rendering=self.rendering_enabled)
        self.client.confirmConnection()
        if self.sim_mode == "Multirotor":
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
        self.connected = True

    def reset(self):
        """Reset the environment and return initial observation"""
        if not self.connected:
            self.connect()

        # Reset drone to initial position
        if self.sim_mode == "Multirotor":
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
        else:
            # For CV mode, just reset pose to origin
            origin_pose = airsim.Pose()
            self.client.simSetVehiclePose(origin_pose, ignore_collision=True)
        self.topdown_map = self.capture_top_down_map(altitude=self.topdown_map_altitude, save_path=None)
        self.topdown_map_with_drone = self.plot_drone_on_topdown_map(self.topdown_map, save_path="true")
        # Return initial observation
        return self._get_observation()

    def step(self, action: Union[np.ndarray, Pose]):
        """Execute action and return observation, reward, done, info"""
        # Action can be either:
        # 1. np.ndarray with [x, y, z, yaw] where yaw is in radians
        # 2. Pose object with x, y, z, and yaw attributes

        if isinstance(action, Pose):
            target_pose = action
        else:
            # If action is array [x, y, z, yaw], convert to Pose
            target_pose = Pose(action[0], action[1], action[2], yaw=action[3])

        self.move_to_pose(target_pose)

        observation = self._get_observation()
        reward = 0.0  # at the moment we dont use any reward
        done = False   # Define your termination condition
        info = {}      # Additional information

        return observation, reward, done, info

    def get_collision_info(self) -> List[str]:
        """Get list of objects the drone has collided with"""
        collision_info = self.client.simGetCollisionInfo()
        collisions = []

        if collision_info.has_collided:
            # Get the object name if available
            object_name = collision_info.object_name if collision_info.object_name else "Unknown"
            collisions.append(object_name)

        return collisions

    def _get_observation(self) -> Observation:
        """Get current observation (image + pose + collisions)"""
        current_pose = self.get_current_pose()
        current_image = self.get_image("0", save_dir=None)  # Don't save during environment steps
        current_collisions = self.get_collision_info()
        current_2dmap = self.plot_drone_on_topdown_map(self.topdown_map, drone_pose=current_pose, save_path=None)
        return Observation(current_image, current_pose, current_collisions, current_2dmap)

    def get_image(self, camera_name, image_type=airsim.ImageType.Scene, save_dir="/media/uam/New Volume/suman/operator-interface/project_logs/imgs"):
        """Return a single image from the specified camera as a PIL image.
        Optionally save the image to save_dir with a unique name if provided."""
        img_bytes = self.client.simGetImage(camera_name, image_type)
        if img_bytes is not None:
            pil_img = Image.open(io.BytesIO(img_bytes))
            if save_dir:
                pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                img_path = pathlib.Path(save_dir) / f"cam_{camera_name}_{current_time}.png"
                pil_img.save(img_path)
            return pil_img
        return None

    def move_to_pose(self, pose: Pose | airsim.Pose, velocity=50):
        # Move to position and set orientation
        if isinstance(pose, airsim.Pose):
                # need to double check the calculation for the yaw angle
                pose = Pose(pose.position.x_val, pose.position.y_val, pose.position.z_val,
                            yaw=np.arctan2(2.0*(pose.orientation.w_val*pose.orientation.z_val + pose.orientation.x_val*pose.orientation.y_val),
                                           1.0 - 2.0*(pose.orientation.y_val**2 + pose.orientation.z_val**2)))
        if self.sim_mode == "Multirotor":
            self.client.moveToPositionAsync(
                pose.x, pose.y, pose.z, velocity
            ).join()

            # Set orientation using roll, pitch, yaw
            if hasattr(pose, 'yaw'):
                self.client.rotateToYawAsync(np.degrees(pose.yaw)).join()
        else:
            # For CV mode, set vehicle pose directly
            airsim_pose = airsim.Pose(
                airsim.Vector3r(pose.x, pose.y, pose.z),
                airsim.to_quaternion(0, 0, pose.yaw)
            )
            self.client.simSetVehiclePose(airsim_pose, ignore_collision=True)

    def get_current_pose(self) -> Pose:
        if self.sim_mode == "Multirotor":
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
        else:
            airsim_pose = self.client.simGetVehiclePose()
            pos = airsim_pose.position
            orientation = airsim_pose.orientation
        # Convert quaternion to Euler angles
        q = orientation
        yaw = np.arctan2(2.0*(q.w_val*q.z_val + q.x_val*q.y_val),
                         1.0 - 2.0*(q.y_val**2 + q.z_val**2))
        pitch = np.arcsin(2.0*(q.w_val*q.y_val - q.z_val*q.x_val))
        roll = np.arctan2(2.0*(q.w_val*q.x_val + q.y_val*q.z_val),
                          1.0 - 2.0*(q.x_val**2 + q.y_val**2))
        return Pose(pos.x_val, pos.y_val, pos.z_val, roll, pitch, yaw)


if __name__ == "__main__":

    ENV_YAML_PATH = "/home/uam/nava/airsim-interface/configs/environments.yaml"
    CONFIG_PATH = "/home/uam/nava/airsim-interface/configs/config.yaml"
    env_name = "ForestEnv"# "AirSimNH" #"ForestEnv"

    def get_active_settings_file(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        sim_mode = config.get("ActiveSimMode", "Multirotor")
        # Hardcoded mapping
        if sim_mode == "Multirotor":
            return str(pathlib.Path(config_path).parent / "settings_multirotor.json")
        elif sim_mode == "ComputerVision":
            return str(pathlib.Path(config_path).parent / "settings_computer_vision_mode.json")
        else:
            raise ValueError(f"Unknown SimMode: {sim_mode}")

    # Usage
    settings_file_path = get_active_settings_file(CONFIG_PATH)

    rendering_enabled = False # Set to False if you want to run without rendering


    unreal_binary = get_unreal_binary(env_name, ENV_YAML_PATH)
    print("Testing AirSimClientWrapper as Gym Environment...")
    image_save_dir = "./project_logs/test_imgs"
    env = None
    try:
        # Try to create environment
        print("Creating AirSim environment...")
        env = AirSimClientWrapper(unreal_binary, rendering_enabled=rendering_enabled, settings_file_path=settings_file_path)

        # Try to connect and reset
        print("Connecting to AirSim...")
        initial_obs = env.reset()
        print("Connected successfully!")
        print(f"Initial observation: {initial_obs}")

        
        # # Now move the drone to demonstrate tracking
        # print("\n=== Moving Drone ===")
        pos_red_car = Pose(x=27.5, y=2.5, z=-3, yaw=180) #env.get_object_pose("Car_10")
        obs, reward, done, info = env.step(pos_red_car)
        obs.image.save(f"./project_logs/test_imgs/step_image.png")
        print(f"Stepped to {pos_red_car}, reward: {reward}, done: {done}, info: {info}")

    except Exception as e:
        print(f"Error during environment setup or testing: {e}")
        print("Attempting to clean up resources...")

    finally:
        # Always try to clean up resources
        if env is not None:
            try:
                print("Stopping Unreal process...")
                env.unreal_manager.stop_process()
                print("Resources cleaned up successfully.")
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")
        else:
            print("No environment to clean up.")

        print("Exiting gracefully.")