from afqa_toolbox.features import FeatOCL, covcoef # type: ignore
import numpy as np

def compute_ocl(self, image, mask):
        
        H, W = image.shape
        ocl_values = []

        blk = self.ocl_blk_size

        for r in range(0, H - blk + 1, blk):
            for c in range(0, W - blk + 1, blk):

                block_mask = mask[r:r+blk, c:c+blk]

                if block_mask.mean() >= self.foreground_ratio:

                    block = image[r:r+blk, c:c+blk]

                    a, b, c_cov_val = covcoef(block, "c_diff_cv")
                    val = FeatOCL.ocl_block(a, b, c_cov_val)

                    ocl_values.append(val)

        if len(ocl_values) == 0:
            return 0.0, 0.0

        ocl_values = np.array(ocl_values, dtype=np.float64)

        return (
            float(np.nanmean(ocl_values))
        )
   