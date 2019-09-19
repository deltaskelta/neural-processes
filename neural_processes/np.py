import torch
from .encoders import Encoder, MuSigmaEncoder, Decoder
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from typing import Tuple, Union, Any
from db.db import DB, str_to_date  # type: ignore
import numpy as np  # type: ignore


class NeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.
    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """

    def __init__(
        self, x_dim: int, y_dim: int, r_dim: int, z_dim: int, h_dim: int
    ) -> None:
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.training: bool

        # Initialize networks
        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    def aggregate(self, r_i: torch.Tensor) -> torch.Tensor:
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.
        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs

        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)  # type: ignore

    def forward(  # type: ignore
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        y_target=None,
    ) -> Union[Normal, Tuple[Normal, ...]]:
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.
        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.
        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.
        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if self.training:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred


class NeuralProcessImg(nn.Module):
    """
    Wraps regular Neural Process for image processing.
    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 28, 28) or (3, 32, 32)
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """

    def __init__(self, img_size: int, r_dim: int, z_dim: int, h_dim: int) -> None:
        super(NeuralProcessImg, self).__init__()
        self.img_size = img_size
        self.num_channels: int = img_size
        self.height: int = img_size
        self.width: int = img_size
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.neural_process = NeuralProcess(
            x_dim=2, y_dim=self.num_channels, r_dim=r_dim, z_dim=z_dim, h_dim=h_dim
        )

    def forward(  # type: ignore
        self,
        img: torch.Tensor,
        context_mask: torch.ByteTensor,
        target_mask: torch.ByteTensor,
    ) -> Union[Normal, Tuple[Normal, ...]]:
        """
        Given an image and masks of context and target points, returns a
        distribution over pixel intensities at the target points.
        Parameters
        ----------
        img : torch.Tensor
            Shape (batch_size, channels, height, width)
        context_mask : torch.ByteTensor
            Shape (batch_size, height, width). Binary mask indicating
            the pixels to be used as context.
        target_mask : torch.ByteTensor
            Shape (batch_size, height, width). Binary mask indicating
            the pixels to be used as target.
        """

        x_context, y_context = img_mask_to_np_input(img, context_mask)
        x_target, y_target = img_mask_to_np_input(img, target_mask)
        return self.neural_process(  # type: ignore (because __call__ on nn.Module returns any)
            x_context, y_context, x_target, y_target
        )  # type: ignore


def context_target_split(
    x: torch.Tensor, y: torch.Tensor, num_context: int, num_extra_target: int
) -> Tuple[torch.Tensor, ...]:
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.
    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)
    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)
    num_context : int
        Number of context points.
    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(
        # num_x_points, size=num_context + num_extra_target, replace=False
        num_points,
        size=num_context,
        replace=False,
    )

    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]

    return x_context, y_context, x_target, y_target


def img_mask_to_np_input(
    img: torch.Tensor, mask: torch.ByteTensor, normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given an image and a mask, return x and y tensors expected by Neural
    Process. Specifically, x will contain indices of unmasked points, e.g.
    [[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
    intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
    [[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.
    Parameters
    ----------
    img : torch.Tensor
        Shape (N, C, H, W). Pixel intensities should be in [0, 1]
    mask : torch.ByteTensor
        Binary matrix where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.
    normalize : bool
        If true normalizes pixel locations x to [-1, 1] and pixel intensities to
        [-0.5, 0.5]
    """
    batch_size, num_channels, height, width = img.size()
    # Create a mask which matches exactly with image size which will be used to
    # extract pixel intensities
    mask_img_size = mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    # Number of points corresponds to number of visible pixels in mask, i.e. sum
    # of non zero indices in a mask (here we assume every mask has same number
    # of visible pixels)
    num_points = mask[0].nonzero().size(0)
    # Compute non zero indices
    # Shape (num_nonzeros, 3), where each row contains index of batch, height and width of nonzero
    nonzero_idx = mask.nonzero()
    # The x tensor for Neural Processes contains (height, width) indices, i.e.
    # 1st and 2nd indices of nonzero_idx (in zero based indexing)
    x = nonzero_idx[:, 1:].view(batch_size, num_points, 2).float()
    # The y tensor for Neural Processes contains the values of non zero pixels
    y = img[mask_img_size].view(batch_size, num_channels, num_points)
    # Ensure correct shape, i.e. (batch_size, num_points, num_channels)
    y = y.permute(0, 2, 1)

    if normalize:
        # TODO: make this separate for height and width for non square image
        # Normalize x to [-1, 1]
        x = (x - float(height) / 2) / (float(height) / 2)
        # Normalize y's to [-0.5, 0.5]
        y -= 0.5

    return x, y


def random_context_target_mask(
    img_size: Tuple[int, ...], num_context: int, num_extra_target: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns random context and target masks where 0 corresponds to a hidden
    value and 1 to a visible value. The visible pixels in the context mask are
    a subset of the ones in the target mask.
    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.
    num_context : int
        Number of context points.
    num_extra_target : int
        Number of additional target points.
    """
    _, height, width = img_size
    # Sample integers without replacement between 0 and the total number of
    # pixels. The measurements array will then contain pixel indices
    # corresponding to locations where pixels will be visible.
    measurements = np.random.choice(
        range(height * width), size=num_context + num_extra_target, replace=False
    )
    # Create empty masks
    context_mask = torch.zeros(width, height).byte()
    target_mask = torch.zeros(width, height).byte()
    # Update mask with measurements
    for i, m in enumerate(measurements):
        row = int(m / width)
        col = m % width
        target_mask[row, col] = 1
        if i < num_context:
            context_mask[row, col] = 1
    return context_mask, target_mask


# TODO: figure out what img_size is and see why it seems to be torhc.Tensor and tuple of ints at the same time
def batch_context_target_mask(
    img_size: Any,
    num_context: int,
    num_extra_target: int,
    batch_size: int,
    repeat: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns bacth of context and target masks, where the visible pixels in
    the context mask are a subset of those in the target mask.
    Parameters
    ----------
    img_size : see random_context_target_mask
    num_context : see random_context_target_mask
    num_extra_target : see random_context_target_mask
    batch_size : int
        Number of masks to create.
    repeat : bool
        If True, repeats one mask across batch.
    """
    context_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    target_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    if repeat:
        context_mask, target_mask = random_context_target_mask(
            img_size, num_context, num_extra_target
        )
        for i in range(batch_size):
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask
    else:
        for i in range(batch_size):
            context_mask, target_mask = random_context_target_mask(
                img_size, num_context, num_extra_target
            )
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask
    return context_mask_batch, target_mask_batch


def xy_to_img(
    x: torch.Tensor, y: torch.Tensor, img_size: Tuple[int, ...]
) -> torch.Tensor:
    """Given an x and y returned by a Neural Process, reconstruct image.
    Missing pixels will have a value of 0.
    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, 2) containing normalized indices.
    y : torch.Tensor
        Shape (batch_size, num_points, num_channels) where num_channels = 1 for
        grayscale and 3 for RGB, containing normalized pixel intensities.
    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.
    """
    _, height, width = img_size
    batch_size, _, _ = x.size()
    # Unnormalize x and y
    x = x * float(height / 2) + float(height / 2)
    x = x.long()
    y += 0.5
    # Permute y so it matches order expected by image
    # (batch_size, num_points, num_channels) -> (batch_size, num_channels, num_points)
    y = y.permute(0, 2, 1)
    # Initialize empty image
    img = torch.zeros((batch_size,) + img_size)
    for i in range(batch_size):
        img[i, :, x[i, :, 0], x[i, :, 1]] = y[i, :, :]
    return img


def inpaint(
    model: NeuralProcessImg,
    img: torch.Tensor,
    context_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Given an image and a set of context points, the model samples pixel
    intensities for the remaining pixels in the image.
    Parameters
    ----------
    model : models.NeuralProcessImg instance
    img : torch.Tensor
        Shape (channels, height, width)
    context_mask : torch.Tensor
        Binary tensor where 1 corresponds to a visible pixel and 0 to an
        occluded pixel. Shape (height, width). Must have dtype=torch.uint8
        or similar. 
    device : torch.device
    """
    is_training = model.neural_process.training
    # For inpainting, use Neural Process in prediction mode
    model.neural_process.training = False
    # All pixels which are not in context
    target_mask: torch.BinaryTensor = 1 - context_mask  # type: ignore (says it is not valid but it actually is)
    # Add a batch dimension to tensors and move to GPU
    img_batch = img.unsqueeze(0).to(device)
    context_batch = context_mask.unsqueeze(0).to(device)
    target_batch = target_mask.unsqueeze(0).to(device)
    p_y_pred = model(img_batch, context_batch, target_batch)
    # Transform Neural Process output back to image
    x_target, _ = img_mask_to_np_input(img_batch, target_batch)
    # Use the mean (i.e. loc) parameter of normal distribution as predictions
    # for y_target
    img_rec = xy_to_img(x_target.cpu(), p_y_pred.loc.detach().cpu(), img.size())
    img_rec = img_rec[0]  # Remove batch dimension
    # Add context points back to image
    context_mask_img = context_mask.unsqueeze(0).repeat(3, 1, 1)
    img_rec[context_mask_img] = img[context_mask_img]
    # Reset model to mode it was in before inpainting
    model.neural_process.training = is_training

    return img_rec
