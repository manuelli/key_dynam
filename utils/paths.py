import os
from key_dynam.utils.utils import get_project_root, get_data_root

SIM_ASSETS_ROOT = os.path.join(get_project_root(), 'sim_assets')
block_push = os.path.join(SIM_ASSETS_ROOT, 'block_push.urdf')
extra_heavy_duty_table = os.path.join(SIM_ASSETS_ROOT, "extra_heavy_duty_table_surface_only_collision.sdf")
xy_slide = os.path.join(SIM_ASSETS_ROOT, "xy_slide.urdf")

ycb_model_paths = dict({
    'cracker_box': os.path.join(SIM_ASSETS_ROOT, "cracker_box/003_cracker_box.sdf"),
    'sugar_box': os.path.join(SIM_ASSETS_ROOT, "sugar_box/004_sugar_box.sdf"),
    'tomato_soup_can': os.path.join(SIM_ASSETS_ROOT, "tomato_soup_can/005_tomato_soup_can.sdf"),
    'mustard_bottle': os.path.join(SIM_ASSETS_ROOT, "mustard_bottle/006_mustard_bottle.sdf"),
    'gelatin_box': os.path.join(SIM_ASSETS_ROOT, "gelatin_box/009_gelatin_box.sdf"),
    'potted_meat_can': os.path.join(SIM_ASSETS_ROOT, "potted_meat_can/010_potted_meat_can.sdf")
})

ycb_model_baselink_names = dict({
    'cracker_box': 'base_link_cracker',
    'sugar_box': 'base_link_sugar',
    'tomato_soup_can': 'base_link_soup',
    'mustard_bottle': 'base_link_mustard',
    'gelatin_box': 'base_link_gelatin',
    'potted_meat_can': 'base_link_meat'
})


LARGE_SIM_ASSETS_ROOT = os.path.join(get_data_root(), 'stable/sim_assets')
