from typing import Any, List, Tuple, Union

import numpy as np  # type: ignore
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

from .encoders import Decoder, Encoder, MuSigmaEncoder


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


def context_target_split(
    x: torch.Tensor,
    y: torch.Tensor,
    num_context: int,
    num_extra_target: int,
    device: torch.device,
    predict_ratio: float = 0.0,
) -> Tuple[torch.Tensor, ...]:
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.
    Parameters
    ----------
    x : Shape (batch_size, num_points, x_dim)
    y : Shape (batch_size, num_points, y_dim)
    num_context : Number of context points.
    num_extra_target :  Number of additional target points.
    predict_ratio: when we want to predict the future, this is the amount of the total we will predict
    """

    num_points = x.shape[1]

    ctx_locations: List[int]
    target_locations: List[int]

    # if we have a predict ratio that means we want to make context all preceeding the target values
    if predict_ratio:
        predict_set_size = int(num_points * predict_ratio)
        ctx_set_size = int(num_points - predict_set_size)

        # an array of locations of training examples for context and target points, contexts are a subset of targets
        # so targets need to concatenate with context and their indexes must only be after the number
        ctx_locations = np.random.choice(ctx_set_size, size=num_context, replace=False)
        target_locations = np.concatenate(
            (
                ctx_locations,
                np.random.choice(predict_set_size, size=num_extra_target, replace=False)
                + ctx_set_size,
            )
        )
    else:
        locations = np.random.choice(
            num_points, size=num_context + num_extra_target, replace=False
        )

        # in this case we don't care about order since we are not trying to predict the latter part of the samples
        # so we can just make context a subset of the target points
        ctx_locations = locations[:num_context]
        target_locations = locations

    x_context = x[:, ctx_locations, :].to(device)
    y_context = y[:, ctx_locations, :].to(device)
    x_target = x[:, target_locations, :].to(device)
    y_target = y[:, target_locations, :].to(device)

    return x_context, y_context, x_target, y_target
