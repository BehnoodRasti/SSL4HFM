from metrics import mse
from metrics import psnr
from metrics import ssim
from metrics import sa
from metrics import L1Error

metrics = {
    "mse": mse.MeanSquaredError,
    "psnr": psnr.PeakSignalToNoiseRatio,
    "ssim":ssim.StructuralSimilarity,
    "sa": sa.SpectralAngle,
    "sa": L1Error.L1Error
}
