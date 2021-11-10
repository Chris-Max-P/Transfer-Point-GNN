from source.data_preprocessing.label_file_operations import *
from source.data_preprocessing.clouds_and_label_splitter import *
from util.paths_and_data import *
def preprocess_data():
    """
    Important: Point-GNN uses a different order of coordinates and corresponding extents:
                l   h   w
    Point-GNN   X   Y   Z
    MUT         Y   Z   X
    Order in label files: see dataset.dataset.get_label()
    """
    # labels =============
    # excel -> .txt files (utf-8)
    change_enc_for_all_files(label_dir)
    #make_Stab_rot_sym(label_dir) # makes posts rotation symmetric by ignoring lower extents
    # change_order_of_label_info(label_dir)  # swap values as needed (primarily for x,y,z,h,w,l)
    # change_label_info('label_dir', 'SR-Stab', 9, '0.1')  # change single values in labels for specific object
        # only needed if not exported correctly from excel
    # setup label configuration splits

    # cloud dir =============
    unpack_directories('source', 'target') # puts all files from subfolders in source into the target folder
    remove_points_from_file_names(tunnel_dir) # removes '.00' from the label file names

    split_all_clouds_and_labels()  # split into smaller pieces for easier computation