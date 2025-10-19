import yaml
import pathlib

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

def get_unreal_binary(env_name, env_yaml_path):
    with open(env_yaml_path, "r") as f:
        envs = yaml.safe_load(f)
    unreal_binary = envs.get(env_name)
    if not unreal_binary:
        raise ValueError(f"Environment '{env_name}' not found in {env_yaml_path}")
    return unreal_binary

# voxel grid related(copied from Taeyoung's work)
import numpy as np

class Voxels(object):
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)
    
def binvox_read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale
    
def binvox_read_as_3d_array(fp, fix_coords=True):
    dims, translate, scale = binvox_read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(bool)
    data = data.reshape(dims)
    if fix_coords:
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)