"""
"fMRI Data Structure" fMRI tutorial in Coursera course fmri003
https://class.coursera.org/fmri-003/wiki/fMRI_Data_Structure_Tutorial

To run this tutorial, download data and unzip into this directory.
https://d396qusza40orc.cloudfront.net/fmri/MoAEpilot.zip
"""
# Created by: Ben Cipollini <bcipolli@ucsd.edu>

import glob
import numpy as np
import os
import matplotlib.pyplot as plt

import nibabel
from nilearn.image import index_img
from nilearn.plotting import plot_anat, plot_stat_map


# Load structural data
print("Loading structural data...")
this_dir = os.path.abspath(os.path.dirname(__file__))
struct_file = os.path.join(this_dir, 'MoAEpilot/sM00223/sM00223_002.img')
struct_img = nibabel.load(struct_file)

# output
print("Structural image size: %s" % str(struct_img.shape))
plot_anat(struct_img, title='Structural image')

# Input the functional volume
print("Loading all functional images into a single volume...")
all_files = glob.glob(os.path.join(this_dir, 'MoAEpilot/fM00223/*.img'))
all_images = [nibabel.funcs.four_to_three(nibabel.load(f))[0]
              for f in all_files]
func_img = nibabel.funcs.concat_images(all_images)

# Define things of interest
z_slice_idx = 39  # 40th image
time_idx = 29  # 30th time point
vox_idx = (19, 19, z_slice_idx)

# output
print("Functional volume size: %s" % str(func_img.shape))
plot_stat_map(index_img(func_img, time_idx), bg_img=struct_img,
              cut_coords=(32, 32, z_slice_idx),
              title='Functional image overlayed on structural image.')
time_data = func_img.get_data()[vox_idx]
fh = plt.figure()
fh.gca().plot(time_data)
fh.gca().set_title('Time series for voxel %s'
                   % str(np.asarray(vox_idx) + 1))  # 1-based idx
plt.show()
