import uuid
from torchvision.utils import save_image
import torch
import torchvision
import numpy as np  # Import numpy for saving weights
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def log_epoch(writer, modus, epoch, loss, mse, psnr, org, rec, target, pred_img, mask_img, model):
    # Log losses / metrics
    writer.add_scalar(modus + "/loss", loss, epoch)
    writer.add_scalar(modus + "/mse", mse, epoch)
    writer.add_scalar(modus + "/psnr", psnr, epoch)

    # Log images
    idx_c = [43, 28, 10]  # r, g, b channels
    writer.add_images(f"{modus}/_org", org[:, idx_c, :, :], epoch, dataformats='NCHW')
    writer.add_images(f"{modus}/_rec", pred_img[:, idx_c, :, :], epoch, dataformats='NCHW')
    writer.add_images(f"{modus}/_target", mask_img[:, idx_c, :, :], epoch, dataformats='NCHW')
    # Initialize a variable to hold the weights
    decoder_linear2_weights = None


    # Attempt to get the weights from the nested path (e.g., for the MAE model)
    if hasattr(model.decoder, 'decoder_linear2'):
        print("Logging nested decoder_linear2 weights")
        # Check if the layer is a container with a 'linear' attribute
        if hasattr(model.decoder.decoder_linear2, 'linear'):
        #     decoder_linear2_weights = model.decoder.decoder_linear2.linear.weight.detach().cpu().numpy()
        # else:
            # Fallback for a direct linear layer within the decoder
            decoder_linear2_weights = model.decoder.decoder_linear2.weight.detach().cpu().numpy()
    elif hasattr(model, 'decoder_linear2'):
        print("Logging direct decoder_linear2 weights")
        # Check if the layer is a container with a 'linear' attribute
        if hasattr(model.decoder_linear2, 'linear'):
        #     decoder_linear2_weights = model.decoder_linear2.linear.weight.detach().cpu().numpy()
        # else:
            # Fallback for a direct linear layer at the top level
            decoder_linear2_weights = model.decoder_linear2.weight.detach().cpu().numpy()
    else:
        print("No decoder_linear2 weights found in the model")
    # Proceed with logging and plotting only if the weights were successfully found
    if decoder_linear2_weights is not None:
        # Extract weights for batch 1, token 1. Note: This assumes 'decoder_linear2_weights'
        # is a 2D array and you want the first row. The name "batch_1_token_1" might be
        # misleading if the tensor represents something else.
        batch_1_token_1_weights = decoder_linear2_weights[0, :]

        # Log histogram and scalar
        writer.add_histogram(f"{modus}/decoder_linear2_weights_batch1_token1", batch_1_token_1_weights, epoch)
        writer.add_scalar(f"{modus}/decoder_linear2_weights_batch1_token1_mean", batch_1_token_1_weights.mean(), epoch)
        
        # Ensure the weights are 2D (e.g., shape [12928, 202])
        decoder_linear2_weights = decoder_linear2_weights.squeeze()

        # Plot the weights
        plt.figure(figsize=(15, 10))
        plt.plot(decoder_linear2_weights, alpha=0.5)
        plt.title(f"Decoder Linear2 Weights - Epoch {epoch}")
        plt.xlabel("Channels (202)")
        plt.ylabel("Weight Value")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to TensorBoard
        writer.add_figure(f"{modus}/decoder_linear2_weights_lines", plt.gcf(), epoch)
        plt.close()
    # # Log weights of decoder_linear2 for batch 1, token 1
    # if hasattr(model, 'decoder_linear2'):  # Ensure the model has the decoder_linear2 attribute
    #     decoder_linear2_weights = model.decoder_conv.weight.detach().cpu()  # Get weights
    #     # Extract weights for batch 1, token 1
    #     batch_1_token_1_weights = decoder_linear2_weights[0, :].numpy().T  # Assuming token 1 corresponds to the first row
    #     writer.add_histogram(f"{modus}/decoder_linear2_weights_batch1_token1", batch_1_token_1_weights, epoch)
    #     writer.add_scalar(f"{modus}/decoder_linear2_weights_batch1_token1_mean", batch_1_token_1_weights.mean(), epoch)
# def log_epoch(writer, modus, epoch, loss,mse, psnr, org, rec, target, pred_img, mask_img):
#     # log losses / metrics
#     writer.add_scalar(modus + "/loss", loss, epoch)
#     # writer.add_scalar(modus + "/bpppc", bpppc, epoch)
#     writer.add_scalar(modus + "/mse", mse, epoch)
#     writer.add_scalar(modus + "/psnr", psnr, epoch)
#     # writer.add_scalar(modus + "/ssim", ssim, epoch)
#     # writer.add_scalar(modus + "/sa", sa, epoch)
#     # log images
#     idx_c = [43, 28, 10]  # r, g, b channels [44, 29, 11]
#     # # log original image batch
#     writer.add_images(f"{modus}/_org", org[:, idx_c, :, :], epoch, dataformats='NCHW')
#     # # log reconstructed image batch
#     writer.add_images(f"{modus}/_rec", pred_img[:, idx_c, :, :], epoch, dataformats='NCHW')
#     writer.add_images(f"{modus}/_target", mask_img[:, idx_c, :, :], epoch, dataformats='NCHW')

