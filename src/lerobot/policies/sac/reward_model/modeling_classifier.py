# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE, REWARD
from lerobot.configs.types import FeatureType


class ClassifierOutput:
    """Wrapper for classifier outputs with additional metadata."""

    def __init__(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
        hidden_states: Tensor | None = None,
    ):
        self.logits = logits
        self.probabilities = probabilities
        self.hidden_states = hidden_states

    def __repr__(self):
        return (
            f"ClassifierOutput(logits={self.logits}, "
            f"probabilities={self.probabilities}, "
            f"hidden_states={self.hidden_states})"
        )


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, H, W, C] or [H, W, C] if no batch
        Returns:
            Output tensor of shape [B, C*F] or [C*F] if no batch
        """

        features = features.last_hidden_state

        original_shape = features.shape
        if features.dim() == 3:
            features = features.unsqueeze(0)  # Add batch dim

        features_expanded = features.unsqueeze(-1)  # [B, H, W, C, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, H, W, C, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum H,W

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        # Remove batch dim
        if len(original_shape) == 3:
            output = output.squeeze(0)

        return output


class Classifier(PreTrainedPolicy):
    """Image classifier built on top of a pre-trained encoder."""

    name = "reward_classifier"
    config_class = RewardClassifierConfig

    def __init__(
        self,
        config: RewardClassifierConfig,
    ):
        from transformers import AutoModel

        super().__init__(config)
        self.config = config

        # Set up encoder
        encoder = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True)
        # Extract vision model if we're given a multimodal model
        if hasattr(encoder, "vision_model"):
            logging.info("Multimodal model detected - using vision encoder only")
            self.encoder = encoder.vision_model
            self.vision_config = encoder.config.vision_config
        else:
            self.encoder = encoder
            self.vision_config = getattr(encoder, "config", None)

        # Model type from config
        self.is_cnn = self.config.model_type == "cnn"

        # For CNNs, initialize backbone
        if self.is_cnn:
            self._setup_cnn_backbone()

        self._freeze_encoder()

        # Extract image keys from input_features
        self.image_keys = [
            key.replace(".", "_") for key in config.input_features if key.startswith(OBS_IMAGE)
        ]

        # Extract state keys (non-visual) from input_features
        self.state_keys = [
            key for key, ft in config.input_features.items() if getattr(ft, "type", None) == FeatureType.STATE
        ]

        if self.is_cnn:
            self.encoders = nn.ModuleDict()
            for image_key in self.image_keys:
                encoder = self._create_single_encoder()
                self.encoders[image_key] = encoder

        # Small MLP to embed state features (if present) to the same latent dim
        if len(self.state_keys) > 0:
            # Use fixed state width from config (strict, no lazy initialization).
            self.state_dim = int(
                sum(math.prod(config.input_features[k].shape) for k in self.state_keys)
            )
            self.state_mlp = nn.Sequential(
                nn.Linear(self.state_dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(self.config.latent_dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        else:
            self.state_dim = 0
            self.state_mlp = None

        self._build_classifier_head()

    def _setup_cnn_backbone(self):
        """Set up CNN encoder"""
        if hasattr(self.encoder, "fc"):
            self.feature_dim = self.encoder.fc.in_features
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        elif hasattr(self.encoder.config, "hidden_sizes"):
            self.feature_dim = self.encoder.config.hidden_sizes[-1]  # Last channel dimension
        else:
            raise ValueError("Unsupported CNN architecture")

    def _freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _create_single_encoder(self):
        encoder = nn.Sequential(
            self.encoder,
            SpatialLearnedEmbeddings(
                height=4,
                width=4,
                channel=self.feature_dim,
                num_features=self.config.image_embedding_pooling_dim,
            ),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.feature_dim * self.config.image_embedding_pooling_dim, self.config.latent_dim),
            nn.LayerNorm(self.config.latent_dim),
            nn.Tanh(),
        )

        return encoder

    def _build_classifier_head(self) -> None:
        """Initialize the classifier head architecture."""
        # Get input dimension based on model type
        if self.is_cnn:
            # image embeddings from all cameras
            image_dim = self.config.latent_dim * self.config.num_cameras
            if self.state_mlp is not None:
                input_dim = image_dim + self.config.latent_dim
            else:
                input_dim = image_dim
        else:  # Transformer models
            if hasattr(self.encoder.config, "hidden_size"):
                input_dim = self.encoder.config.hidden_size
            else:
                raise ValueError("Unsupported transformer architecture since hidden_size is not found")

        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.hidden_dim,
                1 if self.config.num_classes == 2 else self.config.num_classes,
            ),
        )

    def _get_encoder_output(self, x: torch.Tensor, image_key: str) -> torch.Tensor:
        """Extract the appropriate output from the encoder."""
        with torch.no_grad():
            if self.is_cnn:
                # The HF ResNet applies pooling internally
                outputs = self.encoders[image_key](x)
                return outputs
            else:  # Transformer models
                outputs = self.encoder(x)
                return outputs.last_hidden_state[:, 0, :]

    def extract_images_and_labels(self, batch: dict[str, Tensor]) -> tuple[list, Tensor, Tensor | None]:
        """Extract image tensors, label tensors and optional state tensor from batch."""
        # Check for both OBS_IMAGE and OBS_IMAGES prefixes
        images = [batch[key] for key in self.config.input_features if key.startswith(OBS_IMAGE)]
        labels = batch[REWARD]

        state_tensor = None
        if len(self.state_keys) > 0:
            states = [batch[k] for k in self.state_keys]
            # flatten per-state tensors to [B, N]
            states = [s.view(s.size(0), -1) for s in states]
            state_tensor = torch.cat(states, dim=1)
            if state_tensor.shape[1] != self.state_dim:
                raise ValueError(
                    f"State feature dimension mismatch: got {state_tensor.shape[1]} from batch, "
                    f"expected {self.state_dim} from input_features for keys={self.state_keys}. "
                    "Update policy.input_features STATE shapes to match the actual dataset/preprocessor output."
                )

        return images, labels, state_tensor

    def predict(self, xs: list, state: Tensor | None = None) -> ClassifierOutput:
        """Forward pass of the classifier for inference."""
        encoder_outputs = torch.hstack(
            [self._get_encoder_output(x, img_key) for x, img_key in zip(xs, self.image_keys, strict=True)]
        )

        # If state MLP exists, always embed state (use zeros if not provided)
        if self.state_mlp is not None:
            if state is None:
                # Use zero state tensor to maintain expected input dimension
                state = torch.zeros(
                    encoder_outputs.shape[0], self.state_dim, 
                    dtype=encoder_outputs.dtype, device=encoder_outputs.device
                )
            state_emb = self.state_mlp(state)
            encoder_outputs = torch.cat([encoder_outputs, state_emb], dim=1)

        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def _prepare_inference_image(self, image) -> Tensor:
        """Convert env observations to model-ready BCHW float tensors on the model device."""
        tensor = torch.as_tensor(image)

        # Accept single images (HWC or CHW) and batched images (BHWC or BCHW).
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape {tuple(tensor.shape)}")

        # Convert channel-last BHWC to channel-first BCHW when needed.
        if tensor.shape[-1] in (1, 3) and tensor.shape[1] not in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2).contiguous()

        input_dtype = tensor.dtype
        tensor = tensor.to(dtype=torch.float32)
        if input_dtype == torch.uint8:
            tensor /= 255.0

        model_device = next(self.parameters()).device
        return tensor.to(model_device)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Standard forward pass for training compatible with train.py."""
        # Extract images and labels
        images, labels, state = self.extract_images_and_labels(batch)

        # Get predictions (pass state embedding if present)
        outputs = self.predict(images, state)

        # Calculate loss
        if self.config.num_classes == 2:
            # Binary classification
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            # Multi-class classification
            loss = nn.functional.cross_entropy(outputs.logits, labels.long())
            predictions = torch.argmax(outputs.logits, dim=1)

        # Calculate accuracy for logging
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        # Return loss and metrics for logging
        output_dict = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return loss, output_dict

    def predict_reward(self, batch, threshold=0.5):
        """Eval method. Returns predicted reward with the decision threshold as argument."""
        # For online env inference, we may receive raw uint8 HWC frames.
        # Convert them here instead of relying on legacy normalize_* modules.
        images = [
            self._prepare_inference_image(batch[key])
            for key in self.config.input_features
            if key.startswith(OBS_IMAGE)
        ]

        # Prepare state tensor if available in batch.
        # Note: If state_keys are configured but not in batch, predict() will use zeros.
        state = None
        if len(self.state_keys) > 0:
            state_parts = []
            for k in self.state_keys:
                if k in batch:
                    s = torch.as_tensor(batch[k])
                    if s.ndim == 0: # this is done to handle scalar values
                        s = s.unsqueeze(0).unsqueeze(0)
                    elif s.ndim == 1:
                        s = s.unsqueeze(0)
                    s = s.to(dtype=torch.float32)
                    state_parts.append(s.reshape(s.size(0), -1))
                else:
                    logging.info(f"State key '{k}' not in batch; predict() will use zero padding.")
            if len(state_parts) > 0:
                state = torch.cat(state_parts, dim=1).to(next(self.parameters()).device)

        # predict() will handle None state by using zeros if state_mlp exists
        if self.config.num_classes == 2:
            probs = self.predict(images, state).probabilities
            logging.info(f"Predicted reward probs: {probs}")
            return (probs > threshold).float()
        else:
            return torch.argmax(self.predict(images, state).probabilities, dim=1)

    def get_optim_params(self):
        """Return optimizer parameters for the policy."""
        return self.parameters()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not select actions.
        """
        raise NotImplementedError("Reward classifiers do not select actions")

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not produce action chunks.
        """
        raise NotImplementedError("Reward classifiers do not predict action chunks")

    def reset(self):
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not select actions.
        """
        pass
