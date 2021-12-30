#%%
#https://docs.scvi-tools.org/en/stable/tutorials/notebooks/api_overview.html?highlight=clustering#Clustering-on-the-scVI-latent-space
import scvi
import scanpy as sc
import torch

sc.set_figure_params(figsize=(4, 4))
# %%
adata = sc.read_10x_mtx(
    'data/',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)     
# %%
sc.pp.filter_genes(adata, min_counts=3)
adata.layers["counts"] = adata.X.copy() # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata # freeze the state in `.raw`

# %%
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# %%
# scvi.model.LinearSCVI.setup_anndata(adata, layer="counts")

# We make out own model
from scvi._compat import Literal

class MyNeuralNet(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        link_var: Literal["exp", "none", "softmax"],
    ):
        """
        Encodes data of ``n_input`` dimensions into a space of ``n_output`` dimensions.

        Uses a one layer fully-connected neural network with 128 hidden nodes.

        Parameters
        ----------
        n_input
            The dimensionality of the input
        n_output
            The dimensionality of the output
        link_var
            The final non-linearity
        """
        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_output),
        )
        self.transformation = None
        if link_var == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)
        elif link_var == "exp":
            self.transformation = torch.exp

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)
        if self.transformation:
            output = self.transformation(output)
        return output
# %%
import numpy as np
import torch
from torch.distributions import Normal, NegativeBinomial
from torch.distributions import kl_divergence as kl

from scvi import _CONSTANTS
from scvi.module.base import (
    BaseModuleClass,
    LossRecorder,
    auto_move_data,
)

class MyModule(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    n_latent
        Dimensionality of the latent space
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int = 10,
    ):
        super().__init__()
        # in the init, we create the parameters of our elementary stochastic computation unit.

        # First, we setup the parameters of the generative model
        self.decoder = MyNeuralNet(n_latent, n_input, "softmax")
        self.log_theta = torch.nn.Parameter(torch.randn(n_input))

        # Second, we setup the parameters of the variational distribution
        self.mean_encoder = MyNeuralNet(n_input, n_latent, "none")
        self.var_encoder = MyNeuralNet(n_input, n_latent, "exp")

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        # let us fetch the raw counts, and add them to the dictionary
        x = tensors[_CONSTANTS.X_KEY]

        input_dict = dict(x=x)
        return input_dict

    @auto_move_data
    def inference(self, x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1 + x)
        # get variational parameters via the encoder networks
        qz_m = self.mean_encoder(x_)
        qz_v = self.var_encoder(x_)
        # get one sample to feed to the generative model
        # under the hood here is the Reparametrization trick (Rsample)
        z = Normal(qz_m, torch.sqrt(qz_v)).rsample()

        outputs = dict(qz_m=qz_m, qz_v=qz_v, z=z)
        return outputs

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        x = tensors[_CONSTANTS.X_KEY]
        # here we extract the number of UMIs per cell as a known quantity
        library = torch.sum(x, dim=1, keepdim=True)

        input_dict = {
            "z": z,
            "library": library,
        }
        return input_dict

    @auto_move_data
    def generative(self, z, library):
        """Runs the generative model."""

        # get the "normalized" mean of the negative binomial
        px_scale = self.decoder(z)
        # get the mean of the negative binomial
        px_rate = library * px_scale
        # get the dispersion parameter
        theta = torch.exp(self.log_theta)

        return dict(
            px_scale=px_scale, theta=theta, px_rate=px_rate
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):

        # here, we would like to form the ELBO. There are two terms:
        #   1. one that pertains to the likelihood of the data
        #   2. one that pertains to the variational distribution
        # so we extract all the required information
        x = tensors[_CONSTANTS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        theta = generative_outputs["theta"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        # term 1
        # the pytorch NB distribution uses a different parameterization
        # so we must apply a quick transformation (included in scvi-tools, but here we use the pytorch code)
        nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        log_lik = NegativeBinomial(total_counts=theta, total=nb_logits).log_prob(x).sum(dim=-1)

        # term 2
        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence = kl(var_post_dist, prior_dist).sum(dim=1)

        elbo = log_lik - kl_divergence
        loss = torch.mean(-elbo)
        return LossRecorder(loss, -log_lik, kl_divergence, 0.0)


#%%
from typing import Optional

from anndata import AnnData
from scvi.module import VAE
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.data._anndata import _setup_anndata

class SCVI(UnsupervisedTrainingMixin, VAEMixin, BaseModelClass):
    """
    single-cell Variational Inference [Lopez18]_.
    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        **model_kwargs,
    ):
        super(SCVI, self).__init__(adata)

        # self.module = VAE(
        #     n_input=self.summary_stats["n_vars"],
        #     n_batch=self.summary_stats["n_batch"],
        #     n_latent=n_latent,
        #     **model_kwargs,
        # )
        
        self.module = MyModule(n_input=self.summary_stats["n_vars"],
            n_latent=n_latent,
            **model_kwargs
        )


        self._model_summary_string = (
            "SCVI Model with the following params: \nn_latent: {}"
        ).format(
            n_latent,
        )
        self.init_params_ = self._get_init_params(locals())

    @staticmethod
    def setup_anndata(
        adata: AnnData,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        copy: bool = False,
    ) -> Optional[AnnData]:

        return _setup_anndata(
            adata,
            batch_key=batch_key,
            layer=layer,
            copy=copy,
        )


#%%
#model = scvi.model.LinearSCVI(adata, n_latent=10)
SCVI.setup_anndata(adata)
model = SCVI(adata)
# train for 250 epochs, compute metrics every 10 epochs
model.train(max_epochs=20, plan_kwargs={'lr':5e-3}, check_val_every_n_epoch=10)

# %%
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent

# %%
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, key_added="leiden_scVI", resolution=0.5)
# %%
sc.pl.umap(
    adata,
    color=["leiden_scVI"],
    frameon=False,
)
# %%
import pandas as pd
t=adata.X.toarray()
df = pd.DataFrame(data=t, index=adata.obs_names, columns=adata.raw.var_names)
df['cluster'] = adata.obs['leiden_scVI']
#%%
df.to_csv('adata_raw_x.csv')
# %%
