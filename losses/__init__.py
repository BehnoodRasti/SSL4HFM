from losses import mseloss
from losses import ssimloss
from losses import saloss
from losses import GH_Loss   
from losses import L1Loss   
from losses import L1Loss2   
from losses import L1Loss3 
# from losses import con_rec
from losses import MSE_Pos_W  # Import the MSEPosWeightLoss class
from losses import sadloss

losses = {
    "mse": mseloss.MeanSquaredErrorLoss,
    "ssim": ssimloss.StructuralSimilarityLoss,
    "sa": saloss.SpectralAngleLoss,
    "ghmse": GH_Loss.MSEGroupLoss,
    "l1loss": L1Loss.L1ErrorLoss,
    "l1loss2": L1Loss2.L1Error2Loss,
    "l1loss3": L1Loss3.L1SparsLoss,
    # "con_rec": con_rec.ConRecLoss,
    "mse_pos_w": MSE_Pos_W.MSEPosWeightLoss,  # Add the new loss function
    "sad": sadloss.HyperspectralLoss,  # Add the SAD loss function

}
