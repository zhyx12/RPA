from .pascal_voc_format_dataset import *

dataset_name_dict = {
    "pascal_voc_2007": pascal_voc_2007,
    "pascal_voc_2012": pascal_voc_2012,
    "pascal_voc_2007_water": pascal_voc_2007_water,
    "pascal_voc_2012_water": pascal_voc_2012_water,
    "clipart": clipart,
    "water": water,
    "cityscapes": cityscapes,
    "foggy_cityscapes": foggy_cityscapes,
    "cityscapes_car": cityscapes_car,
    "kitti_car": kitti_car,
    "sim10k": sim10k,
    #
    "cityscapes_less1": cityscapes_less1,
    "cityscapes_less2": cityscapes_less2,
    "cityscapes_less3": cityscapes_less3,
    "cityscapes_less4": cityscapes_less4,

    #
    "foggy_cityscapes_car": foggy_cityscapes_car,
    "foggy_cityscapes_less1": foggy_cityscapes_less1,
    "foggy_cityscapes_less2": foggy_cityscapes_less2,
    "foggy_cityscapes_less3": foggy_cityscapes_less3,
    "foggy_cityscapes_less4": foggy_cityscapes_less4,
    #
    'defeat_synthetic': defeat_synthetic,
    'defeat_real': defeat_real,
    #
    'cityscapes_5c': cityscapes_5c,
    'kitti_5c': kitti_5c,
    #
    'cityscapes_7c': cityscapes_7c,
    'bdd100k_7c': bdd100k_7c,
    #
    'cityscapes_from_png': cityscapes_from_png,
    'foggy_cityscapes_from_png': foggy_cityscapes_from_png,
    'cityscapes_car_from_png': cityscapes_car_from_png,
    #
    'cityscapes_from_json': cityscapes_from_json,
    'foggy_cityscapes_from_json': foggy_cityscapes_from_json,
    'cityscapes_car_from_json': cityscapes_car_from_json,
    #
    'cityscapes_5c_from_png': cityscapes_5c_from_png,
    'cityscapes_5c_from_json': cityscapes_5c_from_json,
    #
    'cityscapes_cyclegan': cityscapes_cyclegan,
    'sim10k_cyclegan': sim10k_cyclegan,
    'cityscapes_cyclegan_from_json': cityscapes_cyclegan_from_json,
    #
    'sim10k_munit': sim10k_munit,
    'defeat': Defeat,
}


def get_imdb(name, dataset_params=None):
    """Get an imdb (image database) by name."""
    # if name not in __sets:
    #     raise KeyError('Unknown dataset: {}'.format(name))
    # return __sets[name]()
    dataset_class = dataset_name_dict[name]
    dataset = dataset_class(**dataset_params)
    return dataset
