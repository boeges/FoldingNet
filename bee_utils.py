# Purpose: Defines common util functions and classes for several scripts.

import re
import pandas as pd
from matplotlib import colors
from datetime import datetime
import logging
from pathlib import Path


def Logger(log_file_dir):
  log_file_dir = Path(log_file_dir)
  log_file_dir.mkdir(parents=True, exist_ok=True)
  log_file_path = log_file_dir / (get_date_time_str() + ".log")
  
  # Format
  logging.basicConfig(format="%(asctime)s [%(levelname)-5.5s]  %(message)s")
  logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

  logg = logging.getLogger(__name__)
  logg.setLevel(logging.DEBUG)

  # Log to file (will also log to console)
  fileHandler = logging.FileHandler(log_file_path)
  fileHandler.setFormatter(logFormatter)
  logg.addHandler(fileHandler)

  return logg


# First in tuple MUST be 3-character abbreviation!
CLASS_ABBREVIATIONS = {
    "other":        ("oth","o"),        # 0, other objects that are not insects
    "insect":       ("ins","i"),        # 1, generic/unspecified insect
    "bee":          ("bee","b"),        # 2
    "butterfly":    ("but","u","f"),    # 3
    "dragonfly":    ("dra","d"),        # 4
    "wasp":         ("was","w"),        # 5
    "bumblebee":    ("bum","bb"),       # 6
}

CLASSES = list(CLASS_ABBREVIATIONS.keys())

# scene_id: [full scene name, short scene id]
# TODO remove short ids!
""" scene_id: [full scene name, short scene id] """
SCENE_ID_ALIASES = {
    # HSNR
    "hn-bee-1":         ["hauptsächlichBienen1", "h1"],
    "hn-bee-2":         ["hauptsächlichBienen2", "h2"],
    "hn-dra-1":         ["libellen1", "h3"],
    "hn-dra-2":         ["libellen2", "h4"],
    "hn-dra-3":         ["libellen3", "h5"],
    "hn-but-1":         ["vieleSchmetterlinge1", "h6"],
    "hn-but-2":         ["vieleSchmetterlinge2", "h7"],
    "hn-was-1":         ["wespen1", "h8"],
    "hn-was-2":         ["wespen2", "h9"],
    "hn-was-3":         ["wespen3", "h10"],
    # Muenster
    "mu-1":             ["1_l-l-l", "m1"],
    "mu-2":             ["2_l-h-l", "m2"],
    "mu-3":             ["3_m-h-h", "m3"],
    "mu-4":             ["4_m-m-h", "m4"],
    "mu-5":             ["5_h-l-h", "m5"],
    "mu-6":             ["6_h-h-h_filtered", "m6"],
    # MB
    "mb-dra1-1":        ["mb-dra1-1", None],
    "mb-dra2-1":        ["mb-dra1-1", None],
    "mb-bum2-2":        ["mb-bum2-2", "mb33"],

}
""" short_id: [scene_name, id] """
SCENE_SHORT_ID_ALIASES = {v[1]: [v[0], k] for k, v in SCENE_ID_ALIASES.items()}
""" scene_name: [short_id, id] """
SCENE_NAME_ALIASES = {v[0]: [v[1], k] for k, v in SCENE_ID_ALIASES.items()}

SCENE_IDS = list(SCENE_ID_ALIASES.keys())
SCENE_NAMES = [s[0] for s in SCENE_ID_ALIASES.values()]

# zb "1_l-l-l_trajectories_2024-05-29_15-27-12"
DIR_NAME_PATTERN = re.compile(r"^(.+)_trajectories.*")
# check if a string is a (normal) scene id
SCENE_ID_PATTERN = re.compile(r"^([a-z][a-z0-9]*\-){1,2}[a-z0-9]+$")

CLASS_COLORS = [
    (0.33, 0.33, 0.33, 1.0),        # other, dark grey
    (0.66, 0.66, 0.66, 1.0),        # insect, light grey
    (0.95, 0.45, 0.0, 1.0),         # bee, orange
    (0.6, 0/255, 0.6, 1.0),         # butterfly, purple
    (0.21, 0.49, 0.72, 1.0),        # dragonfly, blue
    (0.95, 0.9, 0.1, 1.0),          # wasp, yellow
    (0.65, 0.33, 0.15, 1.0),        # bumblebee, brown
    (0.30, 0.68, 0.29, 1.0),        # nothing yet, green
    (0.96, 0.50, 0.74, 1.0),        # nothing yet, pink
]

CLASS_CMAP = colors.ListedColormap(CLASS_COLORS)


def get_date_time_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def full_class_to_short_class(cla:str, default=None):
    """ return short name (first 3 characters) """
    return CLASS_ABBREVIATIONS.get(cla.lower(), default[:3])[0]

def parse_full_class_name(cla:str, default=None):
    for c,abbr in CLASS_ABBREVIATIONS.items():
        if cla == c or cla in abbr:
            return c
    return default

def scene_name_to_id(scene_name:str):
    if is_scene_id(scene_name):
        # is already a scene id
        return scene_name
    for id,aliases in SCENE_ID_ALIASES.items():
        if scene_name == id or scene_name in aliases:
            return id
    raise RuntimeError(f"Scene name {scene_name} not in SCENE_ID_ALIASES!")

def scene_short_id_by_id(scene_id:str):
    return scene_aliases_by_id(scene_id)[1]

def scene_aliases_by_id(scene_id:str):
    """ returns: [scene_name, scene_short_id] """
    return SCENE_ID_ALIASES[scene_id]

def scene_aliases_by_short_id(scene_short_id:str):
    """ returns: [scene_name, scene_id ]"""
    return SCENE_SHORT_ID_ALIASES[scene_short_id]
    
def dir_to_scene_name(trajectory_dir_name:str):
    matches = re.findall(DIR_NAME_PATTERN, trajectory_dir_name)
    if len(matches) == 0:
        raise RuntimeError(trajectory_dir_name, " doesnt match pattern!")
    scene_name = matches[0]
    return scene_name

def is_scene_id(s:str):
    return len(re.findall(SCENE_ID_PATTERN, s)) > 0

# make key (scene_id, instance_id, frag_index).
# example: "dragonfly/dragonfly_h3_6_5.csv" becomes ("hn-dra-1", 6, 5).
def frag_filename_to_id(filename) -> tuple:
# get filename
    if "/" in filename or "\\" in filename:
        filename = str(filename).split("/")[-1].split("\\")[-1]
    fn_parts = filename.replace(".csv","").split("_")[-3:]
    frag_id = [None, 0, 0]
    scene_id = fn_parts[0]
    if "-" not in scene_id:
        # is short_id; Convert to normal id: "h3" -> "hn-dra-1"
        scene_id = scene_aliases_by_short_id(scene_id)[1]
    frag_id[0] = scene_id
    frag_id[1] = int(fn_parts[1])
    frag_id[2] = int(fn_parts[2])
    frag_id = tuple(frag_id)
    return frag_id


def frag_filename_to_id_str(fn):
    """
    extracts fragment id string from filename or path string.

    Examples:
    "hn-bee-1_0_17.csv" -> "hn-bee-1_0_17"\n
    "bee/hn-bee-1_0_17.csv" -> "hn-bee-1_0_17"\n
    "bee\\hn-bee-1_0_17.csv" -> "hn-bee-1_0_17"\n

    Args:
        fn (function): _description_

    Returns:
        _type_: _description_
    """
    return "_".join(fn.split("/")[-1].split("\\")[-1].replace(".csv","").split("_")[-3:])


def id_tuple_to_str(id:tuple) -> str:
    return f"{id[0]}_{id[1]}_{id[2]}"

def get_rgba_of_class_index(class_index, alpha=1.0):
    if isinstance(class_index, pd.Series):
        # convert whole series (pd.Series)
        return class_index.apply(lambda v: get_rgba_of_class_index(v, alpha))
    # cnvert single int
    clr = CLASS_COLORS[int(class_index)]
    return tuple([*clr[:3]]+[alpha])
    

def show_colors():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    cmap = CLASS_CMAP

    # Create figure and adjust figure height to number of colormaps
    nrows = 1
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, ax = plt.subplots(nrows=1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    ax.set_title(' colormaps', fontsize=14)

    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.text(-0.01, 0.5, "cmap test", va='center', ha='right', fontsize=10,
            transform=ax.transAxes)

    ax.set_axis_off()

    plt.show()


def read_split_file(split_file_path):
    """
    split file has columns: split [train/test], class name, fragment_id
    fragment id example: hn-bee-1_0_17
    Args:
        split_file_path (_type_): _description_
    """
    with open(split_file_path) as f:
        lines = f.read().splitlines()
        train_fids = []
        test_fids = []
        for line in lines:
            split = line.split(",")[0]
            fid = line.split(",")[2]
            if split == "train":
                train_fids.append(fid)
            elif split == "test":
                test_fids.append(fid)
    return train_fids, test_fids


# DEBUG
if __name__ == "__main__":
    # print("FULL_CLASS_NAMES", CLASSES)
    # print("SCENE_IDS", SCENE_IDS)
    # print("SCENE_NAMES", SCENE_NAMES)
    # print(parse_full_class_name("i", "aaaaa"))

    # print({v[1]: [k, v[0]] for k, v in SCENE_ID_ALIASES.items()})

    # show_colors()

    # print(read_split_file("../../datasets/insect/100ms_4096pts_fps-ds_sor-nr_norm_shufflet_1/train_test_split_2080.txt"))


    print(frag_filename_to_id_str("bee\\hn-bee-1_0_17.csv"))
