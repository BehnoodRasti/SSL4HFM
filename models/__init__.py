from models import mae_vit
from models import aim_vit
from models import mae_gh_vit
from models import mim_vit
models = {
    "mae": mae_vit.mae_vit,
    "aim": aim_vit.aim_vit,
    "mae_GH": mae_gh_vit.mae_gh_vit,
    "mim": mim_vit.mim_vit,
}
