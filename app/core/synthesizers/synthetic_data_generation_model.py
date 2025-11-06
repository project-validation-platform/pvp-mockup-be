import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.errors import NotFittedError, SynthesizerInputError
import os
from app.core.synthesizers.base_model import Generator, Discriminator


class PATEGANSynthesizer(BaseSingleTableSynthesizer):
    def __init__(
        self,
        metadata: SingleTableMetadata,
        latent_dim: int = 128,
        teacher_epochs: int = 100,
        student_epochs: int = 100,
        generator_epochs: int = 100,
        num_teachers: int = 10,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        learning_rate: float = 0.0002,
        batch_size: int = 64,
        teacher_batch_size: int = 32,
        lambda_gradient_penalty: float = 10.0,
        noise_multiplier: float = 1.0,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        locales: Optional[List[str]] = None,
        device: str = 'cpu',
        verbose: bool = True
    ):

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales
        )

        self.latent_dim = latent_dim
        self.teacher_epochs = teacher_epochs
        self.student_epochs = student_epochs
        self.generator_epochs = generator_epochs
        self.num_teachers = num_teachers
        self.epsilon = epsilon
        self.delta = delta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.teacher_batch_size = teacher_batch_size
        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.noise_multiplier = noise_multiplier
        self.verbose = verbose

        self.device = torch.device(
            'cuda' if (device == 'cuda' and torch.cuda.is_available()) else device
        ) if isinstance(device, str) else device

        self.teacher_models: List[Discriminator] = []
        self.student_discriminator: Optional[Discriminator] = None
        self.generator: Optional[Generator] = None
        self.data_dim: Optional[int] = None

        self.criterion = nn.BCELoss()
        self.privacy_spent = 0.0
        self.queries_made = 0

    def _prepare_data(self, data: pd.DataFrame) -> torch.Tensor:
        # Uses BaseSingleTableSynthesizer.preprocess which relies on new Metadata
        transformed = self.preprocess(data)
        self.preprocessed_columns = transformed.columns.tolist()
        return torch.tensor(
            transformed.values, dtype=torch.float32, device=self.device
        )

    def _partition_data(self, data: torch.Tensor) -> List[torch.Tensor]:
        indices = torch.randperm(data.size(0), device=self.device)
        return list(torch.chunk(data[indices], self.num_teachers))

    def _gradient_penalty(self, real, fake, D):
        alpha = torch.rand(real.size(0), 1, device=self.device).expand_as(real)
        interpolates = alpha * real + (1 - alpha) * fake
        interpolates.requires_grad_(True)
        d_interpolates = D(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def _train_teacher(self, data: torch.Tensor) -> Discriminator:
        D = Discriminator(self.data_dim).to(self.device)
        optimizer = optim.Adam(
            D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )

        for _ in range(self.teacher_epochs):
            for i in range(0, data.size(0), self.teacher_batch_size):
                real = data[i:i + self.teacher_batch_size]
                fake = torch.randn_like(real, device=self.device)
                optimizer.zero_grad()
                loss = (
                    -torch.mean(D(real))
                    + torch.mean(D(fake))
                    + self.lambda_gradient_penalty * self._gradient_penalty(real, fake, D)
                )
                loss.backward()
                optimizer.step()

        return D

    def _noisy_vote(self, samples: torch.Tensor) -> torch.Tensor:
        votes = torch.stack([
            (D(samples) > 0).float().squeeze() for D in self.teacher_models
        ])
        vote_sum = votes.sum(dim=0)
        noise = torch.tensor(
            np.random.laplace(0, self.noise_multiplier / self.epsilon, vote_sum.shape),
            dtype=torch.float32, device=self.device
        )
        noisy_votes = vote_sum + noise
        return (noisy_votes > self.num_teachers / 2).float().unsqueeze(1)

    def _train_student(self):
        self.student_discriminator = Discriminator(self.data_dim).to(self.device)
        optimizer = optim.Adam(
            self.student_discriminator.parameters(), lr=self.learning_rate,
            betas=(0.5, 0.999)
        )

        for _ in range(self.student_epochs):
            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z).detach()
            labels = self._noisy_vote(fake)
            optimizer.zero_grad()
            pred = torch.sigmoid(self.student_discriminator(fake))
            loss = self.criterion(pred, labels)
            loss.backward()
            optimizer.step()

    def _train_generator(self):
        optimizer = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate,
            betas=(0.5, 0.999)
        )

        for _ in range(self.generator_epochs):
            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z)
            pred = self.student_discriminator(fake)
            loss = -torch.mean(pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fit(self, data: pd.DataFrame):
        if data.empty:
            raise SynthesizerInputError("Input data is empty.")

        tensor_data = self._prepare_data(data)
        self.data_dim = tensor_data.shape[1]
        self.generator = Generator(self.latent_dim, self.data_dim).to(self.device)

        partitions = self._partition_data(tensor_data)
        self.teacher_models = [self._train_teacher(p) for p in partitions]
        self._train_student()
        self._train_generator()

        self._fitted = True

    def _sample(self, num_rows: int, conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if not self._fitted:
            raise NotFittedError("PATEGANSynthesizer must be fitted before sampling.")

        self.generator.eval()
        all_batches = []
        batches_required = (num_rows + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for _ in range(batches_required):
                z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
                fake = self.generator(z)
                needed = min(self.batch_size, num_rows - len(all_batches) * self.batch_size)
                all_batches.append(fake[:needed])

        synthetic_array = torch.cat(all_batches, dim=0).cpu().numpy()
        synthetic_df = pd.DataFrame(synthetic_array, columns=self.preprocessed_columns)
        self._data_processor.reverse_transform(synthetic_df).to_csv(os.path.join("results", "test.csv"))
        return self._data_processor.reverse_transform(synthetic_df)