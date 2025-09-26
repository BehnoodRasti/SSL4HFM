import torch
import torch.nn as nn
import torch.nn.functional as F
class ImprovedSADLoss(nn.Module):
    """
    Improved SAD Loss with better convergence properties for hyperspectral MAE
    """
    
    def __init__(self, reduction='mean', eps=1e-8, expected_bands=202, 
                 scale_factor=1.0, use_degrees=False, combined_loss=False, 
                 mse_weight=0.7, sad_weight=0.3):
        super(ImprovedSADLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.expected_bands = expected_bands
        self.scale_factor = scale_factor
        self.use_degrees = use_degrees  # Convert to degrees for larger gradients
        self.combined_loss = combined_loss  # Combine with MSE loss
        self.mse_weight = mse_weight
        self.sad_weight = sad_weight
        
        if combined_loss:
            self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target):
        """
        Improved forward pass with debugging and better convergence
        """
        # Ensure inputs have the same shape
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        
        # Debug: Check input statistics
        with torch.no_grad():
            pred_mean = pred.mean().item()
            target_mean = target.mean().item()
            pred_std = pred.std().item()
            target_std = target.std().item()
            
            if abs(pred_mean) > 100 or abs(target_mean) > 100:
                print(f"Warning: Large input values detected - pred_mean: {pred_mean:.3f}, target_mean: {target_mean:.3f}")
            
            if pred_std < 1e-6 or target_std < 1e-6:
                print(f"Warning: Very small std detected - pred_std: {pred_std:.6f}, target_std: {target_std:.6f}")
        
        # Handle the reshape logic for MAE patches
        if pred.dim() == 3:
            batch_size, dim1, dim2 = pred.shape
            
            if dim2 > self.expected_bands and dim2 % self.expected_bands == 0:
                pixels_per_patch = dim2 // self.expected_bands
                pred = pred.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
                target = target.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
            elif dim2 == self.expected_bands:
                pass
            elif dim1 == self.expected_bands:
                pred = pred.transpose(1, 2)
                target = target.transpose(1, 2)
            else:
                raise ValueError(f"Cannot determine bands dimension from shape {pred.shape}")
        
        batch_size, num_pixels, bands = pred.shape
        pred_flat = pred.reshape(-1, bands)
        target_flat = target.reshape(-1, bands)
        
        # Normalize spectra to unit vectors (important for SAD)
        pred_norm = F.normalize(pred_flat, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(target_flat, p=2, dim=1, eps=self.eps)
        
        # Compute cosine similarity (dot product of normalized vectors)
        cos_similarity = torch.sum(pred_norm * target_norm, dim=1)
        
        # More robust clamping
        cos_similarity = torch.clamp(cos_similarity, -1.0 + self.eps, 1.0 - self.eps)
        
        # Compute spectral angle
        spectral_angles = torch.acos(cos_similarity)
        
        # Convert to degrees if requested (larger gradients)
        if self.use_degrees:
            spectral_angles = spectral_angles * 180.0 / torch.pi
        
        # Apply scale factor
        spectral_angles = spectral_angles * self.scale_factor
        
        # Debug: Check loss statistics
        with torch.no_grad():
            angle_mean = spectral_angles.mean().item()
            angle_std = spectral_angles.std().item()
            angle_max = spectral_angles.max().item()
            if angle_mean > 1.0 or angle_std > 1.0:
                print(f"SAD stats - mean: {angle_mean:.6f}, std: {angle_std:.6f}, max: {angle_max:.6f}")
        
        # Combine with MSE loss if requested
        if self.combined_loss:
            mse_loss = self.mse_loss(pred, target)
            sad_loss = self._apply_reduction(spectral_angles, batch_size, num_pixels)
            
            # Debug combined loss
            with torch.no_grad():
                print(f"MSE: {mse_loss.item():.6f}, SAD: {sad_loss.item():.6f}")
            
            return self.mse_weight * mse_loss + self.sad_weight * sad_loss
        
        return self._apply_reduction(spectral_angles, batch_size, num_pixels)
    
    def _apply_reduction(self, spectral_angles, batch_size, num_pixels):
        """Apply reduction with proper reshaping"""
        if self.reduction == 'mean':
            return spectral_angles.mean()
        elif self.reduction == 'sum':
            return spectral_angles.sum()
        elif self.reduction == 'none':
            return spectral_angles.reshape(batch_size, num_pixels)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class HyperspectralLoss(nn.Module):
    """
    Multi-component loss specifically designed for hyperspectral reconstruction
    """
    
    def __init__(self, mse_weight=0.6, sad_weight=0.3, l1_weight=0.1, 
                 expected_bands=202):
        super(HyperspectralLoss, self).__init__()
        self.mse_weight = mse_weight
        self.sad_weight = sad_weight
        self.l1_weight = l1_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.sad_loss = ImprovedSADLoss(expected_bands=expected_bands, 
                                       scale_factor=10.0, use_degrees=True)
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        sad = self.sad_loss(pred, target)
        
        total_loss = (self.mse_weight * mse + 
                     #self.l1_weight * l1 + 
                     0.1*self.sad_weight * sad)
        
        # Debug output
        with torch.no_grad():
            print(f"Loss components - MSE: {mse.item():.6f}, L1: {l1.item():.6f}, "
                  f"SAD: {sad.item():.6f}, Total: {total_loss.item():.6f}")
        
        return total_loss


def debug_model_outputs(pred, target, loss_fn):
    """
    Debug function to analyze model outputs and loss behavior
    """
    print("="*50)
    print("DEBUGGING MODEL OUTPUTS")
    print("="*50)
    
    with torch.no_grad():
        # Input statistics
        print(f"Pred shape: {pred.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Pred - min: {pred.min():.6f}, max: {pred.max():.6f}, mean: {pred.mean():.6f}, std: {pred.std():.6f}")
        print(f"Target - min: {target.min():.6f}, max: {target.max():.6f}, mean: {target.mean():.6f}, std: {target.std():.6f}")
        
        # Difference statistics
        diff = pred - target
        print(f"Diff - min: {diff.min():.6f}, max: {diff.max():.6f}, mean: {diff.mean():.6f}, std: {diff.std():.6f}")
        
        # Loss values
        loss = loss_fn(pred, target)
        mse_loss = F.mse_loss(pred, target)
        l1_loss = F.l1_loss(pred, target)
        
        print(f"MSE Loss: {mse_loss.item():.6f}")
        print(f"L1 Loss: {l1_loss.item():.6f}")
        print(f"Current Loss: {loss.item():.6f}")
        
        # PSNR calculation
        mse_val = mse_loss.item()
        if mse_val > 0:
            # Assuming pixel values are in range [0, 1]
            psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse_val))
            print(f"PSNR: {psnr.item():.2f} dB")
        else:
            print("PSNR: inf (perfect reconstruction)")
        
        # Check for common issues
        if torch.isnan(pred).any():
            print("WARNING: NaN values in predictions!")
        if torch.isinf(pred).any():
            print("WARNING: Inf values in predictions!")
        if pred.requires_grad and pred.grad is not None:
            print(f"Pred grad - min: {pred.grad.min():.6f}, max: {pred.grad.max():.6f}")


# Example usage and debugging
if __name__ == "__main__":
    # Test with your dimensions
    batch_size = 64
    visible_patches = 58
    bands = 202
    pixels_per_patch = 256
    flattened_size = pixels_per_patch * bands
    
    # Create test data
    pred = torch.randn(batch_size, visible_patches, flattened_size, requires_grad=True)
    target = torch.randn(batch_size, visible_patches, flattened_size)
    
    # Test different loss configurations
    print("Testing Standard SAD Loss:")
    sad_loss = ImprovedSADLoss()
    debug_model_outputs(pred, target, sad_loss)
    
    print("\nTesting Combined Loss:")
    combined_loss = ImprovedSADLoss(combined_loss=True, mse_weight=0.7, sad_weight=0.3)
    debug_model_outputs(pred, target, combined_loss)
    
    print("\nTesting Multi-component Loss:")
    multi_loss = HyperspectralLoss()
    debug_model_outputs(pred, target, multi_loss)
# class SADLoss(nn.Module):
#     """
#     Spectral Angle Distance (SAD) Loss Function for Hyperspectral MAE
    
#     Computes the spectral angle between predicted and target spectra.
#     Designed for hyperspectral images with 202 bands processed through MAE.
    
#     Args:
#         reduction (str): Specifies the reduction to apply to the output:
#                         'mean' | 'sum' | 'none'. Default: 'mean'
#         eps (float): Small value to avoid division by zero. Default: 1e-8
#         expected_bands (int): Expected number of spectral bands. Default: 202
#     """
    
#     def __init__(self, reduction='mean', eps=1e-8, expected_bands=202):
#         super(SADLoss, self).__init__()
#         self.reduction = reduction
#         self.eps = eps
#         self.expected_bands = expected_bands
    
#     def forward(self, pred, target):
#         """
#         Args:
#             pred: Predicted hyperspectral data from MAE. Expected shapes:
#                   - (batch_size, visible_patches, flattened_patch_data) -> [64, 58, 51712]
#                   - (batch_size, num_pixels, bands) -> [64, pixels, 202]
#                   - (batch_size, bands, num_pixels) -> [64, 202, pixels]
#             target: Target hyperspectral data of same shape as pred
        
#         Returns:
#             SAD loss value
#         """
#         # Ensure inputs have the same shape
#         assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        
#         # Handle the case where data might be flattened patches from MAE
#         if pred.dim() == 3:
#             batch_size, dim1, dim2 = pred.shape
            
#             # Case 1: [64, 58, 51712] - MAE patches with flattened data
#             if dim2 > self.expected_bands and dim2 % self.expected_bands == 0:
#                 # Reshape to separate pixels and bands
#                 pixels_per_patch = dim2 // self.expected_bands
#                 # Reshape to [batch_size, visible_patches * pixels_per_patch, bands]
#                 pred = pred.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
#                 target = target.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
                
#             # Case 2: [batch_size, num_pixels, bands]
#             elif dim2 == self.expected_bands:
#                 # Already in correct format
#                 pass
                
#             # Case 3: [batch_size, bands, num_pixels] 
#             elif dim1 == self.expected_bands:
#                 # Transpose to put bands last
#                 pred = pred.transpose(1, 2)
#                 target = target.transpose(1, 2)
                
#             else:
#                 # Try to auto-detect by checking if any dimension equals expected_bands
#                 if dim1 == self.expected_bands:
#                     pred = pred.transpose(1, 2)
#                     target = target.transpose(1, 2)
#                 elif dim2 % self.expected_bands == 0:
#                     # Assume flattened format
#                     pixels_per_patch = dim2 // self.expected_bands
#                     pred = pred.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
#                     target = target.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
#                 else:
#                     raise ValueError(f"Cannot determine bands dimension from shape {pred.shape}. "
#                                    f"Expected to find dimension with size {self.expected_bands} or "
#                                    f"a dimension divisible by {self.expected_bands}")
        
#         # Now we should have shape [batch_size, num_pixels, bands]
#         batch_size, num_pixels, bands = pred.shape
        
#         # Reshape to [batch_size * num_pixels, bands] for pixel-wise computation
#         pred_flat = pred.reshape(-1, bands)
#         target_flat = target.reshape(-1, bands)
        
#         # Compute dot product between corresponding spectra
#         dot_product = torch.sum(pred_flat * target_flat, dim=1)
        
#         # Compute L2 norms
#         pred_norm = torch.norm(pred_flat, dim=1) + self.eps
#         target_norm = torch.norm(target_flat, dim=1) + self.eps
        
#         # Compute cosine similarity
#         cos_similarity = dot_product / (pred_norm * target_norm)
        
#         # Clamp to avoid numerical issues with arccos
#         cos_similarity = torch.clamp(cos_similarity, -1.0 + self.eps, 1.0 - self.eps)
        
#         # Compute spectral angle (in radians)
#         spectral_angles = torch.acos(cos_similarity)
        
#         # Apply reduction
#         if self.reduction == 'mean':
#             return spectral_angles.mean()
#         elif self.reduction == 'sum':
#             return spectral_angles.sum()
#         elif self.reduction == 'none':
#             # Reshape back to [batch_size, num_pixels]
#             return spectral_angles.reshape(batch_size, num_pixels)
#         else:
#             raise ValueError(f"Invalid reduction mode: {self.reduction}")


# # Alternative implementation using cosine similarity directly
# class SADLossV2(nn.Module):
#     """
#     Alternative SAD Loss using PyTorch's cosine similarity
#     More efficient implementation
#     """
    
#     def __init__(self, reduction='mean', eps=1e-8, expected_bands=202):
#         super(SADLossV2, self).__init__()
#         self.reduction = reduction
#         self.eps = eps
#         self.expected_bands = expected_bands
    
#     def forward(self, pred, target):
#         # Same reshaping logic as SADLoss
#         if pred.dim() == 3:
#             batch_size, dim1, dim2 = pred.shape
            
#             if dim2 > self.expected_bands and dim2 % self.expected_bands == 0:
#                 pixels_per_patch = dim2 // self.expected_bands
#                 pred = pred.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
#                 target = target.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
#             elif dim2 == self.expected_bands:
#                 pass
#             elif dim1 == self.expected_bands:
#                 pred = pred.transpose(1, 2)
#                 target = target.transpose(1, 2)
#             else:
#                 if dim1 == self.expected_bands:
#                     pred = pred.transpose(1, 2)
#                     target = target.transpose(1, 2)
#                 elif dim2 % self.expected_bands == 0:
#                     pixels_per_patch = dim2 // self.expected_bands
#                     pred = pred.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
#                     target = target.reshape(batch_size, dim1 * pixels_per_patch, self.expected_bands)
#                 else:
#                     raise ValueError(f"Cannot determine bands dimension from shape {pred.shape}")
        
#         batch_size, num_pixels, bands = pred.shape
#         pred_flat = pred.reshape(-1, bands)
#         target_flat = target.reshape(-1, bands)
        
#         # Compute cosine similarity
#         cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=self.eps)
        
#         # Clamp and compute spectral angle
#         cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
#         spectral_angles = torch.acos(cos_sim)
        
#         # Apply reduction
#         if self.reduction == 'mean':
#             return spectral_angles.mean()
#         elif self.reduction == 'sum':
#             return spectral_angles.sum()
#         elif self.reduction == 'none':
#             return spectral_angles.reshape(batch_size, num_pixels)
#         else:
#             raise ValueError(f"Invalid reduction mode: {self.reduction}")


# # Example usage
# if __name__ == "__main__":
#     # Example with your MAE setup
#     batch_size = 64
#     visible_patches = 58  # After masking in MAE
#     patch_size = 16
#     bands = 202
    
#     # Case 1: Flattened patches [64, 58, 51712]
#     # If 51712 = 58 patches × 256 pixels/patch × some factor
#     # Let's assume 51712 = 256 * 202 = pixels_per_patch * bands
#     pixels_per_patch = 256  # 16×16
#     flattened_size = pixels_per_patch * bands  # 256 * 202 = 51712
    
#     pred = torch.randn(batch_size, visible_patches, flattened_size)
#     target = torch.randn(batch_size, visible_patches, flattened_size)
    
#     # Initialize SAD loss
#     sad_loss = SADLoss(reduction='mean')
    
#     # Compute loss
#     loss = sad_loss(pred, target)
#     print(f"SAD Loss (MAE patches): {loss.item():.6f}")
    
#     # Case 2: Standard format [batch_size, num_pixels, bands]
#     total_pixels = 128 * 128  # Full image
#     pred_std = torch.randn(batch_size, total_pixels, bands)
#     target_std = torch.randn(batch_size, total_pixels, bands)
    
#     loss_std = sad_loss(pred_std, target_std)
#     print(f"SAD Loss (standard format): {loss_std.item():.6f}")
    
#     # Test with no reduction
#     sad_loss_none = SADLoss(reduction='none')
#     loss_map = sad_loss_none(pred, target)
#     print(f"Loss map shape: {loss_map.shape}")
    
#     print(f"\nExpected input shapes:")
#     print(f"- MAE patches: [64, 58, {flattened_size}] -> pixels per patch: {pixels_per_patch}")
#     print(f"- Standard: [64, {total_pixels}, 202]")
#     print(f"- Bands first: [64, 202, {total_pixels}]")