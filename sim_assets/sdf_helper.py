import glob
import os
import shutil
import numpy as np
from PIL import Image

from key_dynam.utils.utils import get_current_YYYY_MM_DD_hh_mm_ss_ms

class SDFHelper(object):

    @staticmethod
    def get_sdf_list(sdf_dir):
        """
        Returns a list of sdfs in that directory
        :param sdf_dir:
        :type sdf_dir:
        :return:
        :rtype:
        """
        pattern = sdf_dir + "/*.sdf"
        file_list = glob.glob(pattern)
        file_list.sort()
        return file_list


    @staticmethod
    def create_sdf_specific_color(sdf_file_fullpath, # e.g. /home/user/model.sdf
                                  color, # [r, g, b] e.g. [255, 0, 0] etc.
                                  output_dir=None):

        if output_dir is None:
            output_dir = os.path.join("/tmp/sdf_helper", get_current_YYYY_MM_DD_hh_mm_ss_ms())

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sdf_dir = os.path.dirname(sdf_file_fullpath)
        sdf_file = os.path.basename(sdf_file_fullpath)
        sdf_file_dst = os.path
        basename = os.path.splitext(sdf_file)[0]

        # copy all files that match basename* to the output dir
        pattern = os.path.join(sdf_dir, basename) + "*"

        for filename in glob.iglob(pattern):
            dst = os.path.join(output_dir, os.path.basename(filename))
            shutil.copyfile(filename, dst)



        img = np.uint8(np.ones([10,10, 3], dtype=np.uint8) * np.array(color))
        img_PIL = Image.fromarray(img)
        img_PIL.save(os.path.join(output_dir, basename + ".png"))
        return {'dir': output_dir,
                'basename': basename,
                'sdf_file': os.path.join(output_dir, basename + ".sdf"),
                }



def test():
    print("test")
    output_dir = "/home/manuelli/data/key_dynam/sandbox/sdf_helper"
    sdf_file_fullpath = "/home/manuelli/data/key_dynam/stable/sim_assets/anzu_mugs/big_mug-corelle_mug-0.obj"

    color = [255, 0, 0]
    SDFHelper.create_sdf_specific_color(sdf_file_fullpath, color, output_dir=output_dir)


if __name__ == "__main__":
    test()