from source.dataset_classes.point_cloud_processing import display_cloud_by_filename
from util.paths_and_data import *

file = '5-2_546700_546800(9)'
filename = os.path.join(tunnel_dir, file + '.e57')
display_cloud_by_filename(filename)