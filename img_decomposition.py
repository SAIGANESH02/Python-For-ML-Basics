import os
import numpy as np
import image_util
# from numba import vectorize,jitclass,jit

class IntrinsicDecomposition(object):
    """ Current state of a reconstruction.  All entries (except ``input``) are
    mutable. """

    def __init__(self, params, input):
        self._input = input
        self.params = params

        # iteration number
        self.iter_num = None

        # stage 1 or 2 (each iteration has 2 stages)
        self.stage_num = None

        # labels ("x" variable in the paper), where "_nz" indicates that only the
        # nonmasked entries are stored.
        self.labels_nz = None

        # reflectance intensity (obtained from kmeans)
        self.intensities = None
        # reflectance chromaticity (obtained from kmeans)
        self.chromaticities = None

        # store here for visualization only
        self.shading_target = None
def copy(self):
        ret = IntrinsicDecomposition(self.params, self.input)
        ret.iter_num = self.iter_num
        ret.stage_num = self.stage_num
        ret.labels_nz = self.labels_nz.copy()
        ret.intensities = self.intensities.copy()
        ret.chromaticities = self.chromaticities.copy()
        if self.shading_target is not None:
            ret.shading_target = self.shading_target.copy()
        return ret

    def get_r_s_nz(self):
        """ Return (reflectance, shading), with just the nonmasked entries """
        s_nz = self.input.image_gray_nz / self.intensities[self.labels_nz]
        r_nz = self.input.image_rgb_nz / np.clip(s_nz, 1e-4, 1e5)[:, np.newaxis]
        assert s_nz.ndim == 1 and r_nz.ndim == 2 and r_nz.shape[1] == 3
        return r_nz, s_nz
