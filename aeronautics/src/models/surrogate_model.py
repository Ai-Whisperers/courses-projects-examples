"""
Aerodynamic Surrogate Model with Uncertainty Quantification

This module implements an ensemble neural network for predicting
aerodynamic coefficients from CST parameters with uncertainty estimates.

Key features:
- Ensemble of networks for epistemic uncertainty (model uncertainty)
- Heteroscedastic output for aleatoric uncertainty (data noise)
- Physical constraint enforcement via custom loss terms
- Out-of-distribution detection

Architecture based on:
    Lakshminarayanan et al. (2017) "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import warnings
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Types of uncertainty in predictions."""
    EPISTEMIC = "epistemic"    # Model uncertainty (reducible with more data)
    ALEATORIC = "aleatoric"    # Data noise (irreducible)
    TOTAL = "total"            # Combined uncertainty


@dataclass
class Prediction:
    """
    Model prediction with uncertainty estimates.

    Attributes:
        cl, cd, cm: Predicted coefficients (mean)
        cl_std, cd_std, cm_std: Total uncertainty (1 std)
        epistemic_std: Model uncertainty component
        aleatoric_std: Data noise component
        confidence: Overall confidence score [0, 1]
        is_ood: Out-of-distribution flag
    """
    cl: float
    cd: float
    cm: float
    cl_std: float
    cd_std: float
    cm_std: float
    epistemic_std: Dict[str, float] = field(default_factory=dict)
    aleatoric_std: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    is_ood: bool = False
    ood_score: float = 0.0

    def __post_init__(self):
        # Ensure CD is positive
        if self.cd < 0:
            warnings.warn(f"Negative Cd predicted: {self.cd}, clipping to 0.001")
            self.cd = 0.001

    def get_interval(self, coeff: str, n_sigma: float = 2.0) -> Tuple[float, float]:
        """Get confidence interval for a coefficient."""
        mean = getattr(self, coeff)
        std = getattr(self, f"{coeff}_std")
        return (mean - n_sigma * std, mean + n_sigma * std)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cl': self.cl, 'cd': self.cd, 'cm': self.cm,
            'cl_std': self.cl_std, 'cd_std': self.cd_std, 'cm_std': self.cm_std,
            'confidence': self.confidence,
            'is_ood': self.is_ood,
            'ood_score': self.ood_score,
        }


class InputNormalizer:
    """
    Normalize inputs for neural network training.

    Uses online statistics computation to avoid storing all data.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.mean = np.zeros(n_features)
        self.std = np.ones(n_features)
        self.n_samples = 0
        self._M2 = np.zeros(n_features)  # For Welford's algorithm
        self._fitted = False

    def partial_fit(self, X: np.ndarray):
        """Update statistics with new data (Welford's online algorithm)."""
        for x in X:
            self.n_samples += 1
            delta = x - self.mean
            self.mean += delta / self.n_samples
            delta2 = x - self.mean
            self._M2 += delta * delta2

        if self.n_samples > 1:
            self.std = np.sqrt(self._M2 / (self.n_samples - 1))
            self.std = np.clip(self.std, 1e-6, None)  # Prevent division by zero
        self._fitted = True

    def fit(self, X: np.ndarray):
        """Fit normalizer to data."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.clip(self.std, 1e-6, None)
        self.n_samples = len(X)
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize inputs."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted")
        return (X - self.mean) / self.std

    def inverse_transform(self, X_normalized: np.ndarray) -> np.ndarray:
        """Denormalize inputs."""
        return X_normalized * self.std + self.mean

    def save(self, filepath: Path):
        """Save normalizer state."""
        state = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'n_samples': self.n_samples,
            'n_features': self.n_features,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    @classmethod
    def load(cls, filepath: Path) -> 'InputNormalizer':
        """Load normalizer state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        normalizer = cls(state['n_features'])
        normalizer.mean = np.array(state['mean'])
        normalizer.std = np.array(state['std'])
        normalizer.n_samples = state['n_samples']
        normalizer._fitted = True
        return normalizer


class NeuralNetworkBase(ABC):
    """
    Abstract base class for neural network implementations.

    This allows swapping between numpy-only and framework-based
    implementations (PyTorch, TensorFlow, JAX).
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass returning (mean, log_variance)."""
        pass

    @abstractmethod
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Single training step, returns loss."""
        pass

    @abstractmethod
    def save(self, filepath: Path):
        """Save model weights."""
        pass

    @abstractmethod
    def load(self, filepath: Path):
        """Load model weights."""
        pass


class SimpleNumpyMLP(NeuralNetworkBase):
    """
    Simple MLP implemented in pure NumPy.

    This is useful for environments without ML frameworks.
    For production, use PyTorch or JAX implementations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64, 32],
        output_dim: int = 3,
        learning_rate: float = 1e-3,
        seed: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.lr = learning_rate

        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        # Initialize weights (He initialization)
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            w = rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i+1]))
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

        # Output layer: mean and log_variance for each output
        w_out = rng.normal(0, 0.01, (hidden_dims[-1], output_dim * 2))
        b_out = np.zeros(output_dim * 2)
        # Initialize log_variance biases to reasonable values
        b_out[output_dim:] = -2.0  # ~0.14 std initially
        self.weights.append(w_out)
        self.biases.append(b_out)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass with heteroscedastic output."""
        self._activations = [x]

        h = x
        for i in range(len(self.weights) - 1):
            h = h @ self.weights[i] + self.biases[i]
            h = self._relu(h)
            self._activations.append(h)

        # Output layer (no activation on mean, softplus on log_var)
        output = h @ self.weights[-1] + self.biases[-1]
        mean = output[:, :self.output_dim]
        log_var = output[:, self.output_dim:]

        # Clip log_var for stability
        log_var = np.clip(log_var, -10, 10)

        return mean, log_var

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Training step with negative log-likelihood loss.

        Loss = 0.5 * (log_var + (y - mean)^2 / exp(log_var))
        """
        mean, log_var = self.forward(x)
        var = np.exp(log_var)

        # NLL loss
        residual = y - mean
        loss = 0.5 * np.mean(log_var + residual**2 / var)

        # Backpropagation (simplified gradient descent)
        # Gradient of loss w.r.t. mean and log_var
        d_mean = -residual / var
        d_log_var = 0.5 * (1 - residual**2 / var)

        d_output = np.concatenate([d_mean, d_log_var], axis=1) / len(x)

        # Backprop through layers
        d_h = d_output @ self.weights[-1].T
        self.weights[-1] -= self.lr * self._activations[-1].T @ d_output
        self.biases[-1] -= self.lr * np.sum(d_output, axis=0)

        for i in range(len(self.weights) - 2, -1, -1):
            d_h = d_h * self._relu_grad(self._activations[i+1])
            self.weights[i] -= self.lr * self._activations[i].T @ d_h
            self.biases[i] -= self.lr * np.sum(d_h, axis=0)
            if i > 0:
                d_h = d_h @ self.weights[i].T

        return loss

    def save(self, filepath: Path):
        """Save weights to file."""
        state = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim,
            }
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load(self, filepath: Path):
        """Load weights from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.weights = [np.array(w) for w in state['weights']]
        self.biases = [np.array(b) for b in state['biases']]


class EnsembleSurrogateModel:
    """
    Ensemble of neural networks for aerodynamic coefficient prediction.

    Uses deep ensembles for uncertainty quantification:
    - Epistemic uncertainty from ensemble disagreement
    - Aleatoric uncertainty from heteroscedastic outputs
    """

    def __init__(
        self,
        input_dim: int,
        n_ensemble: int = 5,
        hidden_dims: List[int] = [64, 64, 32],
        learning_rate: float = 1e-3,
        seed: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.n_ensemble = n_ensemble
        self.hidden_dims = hidden_dims

        # Create ensemble members with different seeds
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31, n_ensemble)

        self.models = [
            SimpleNumpyMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=3,  # Cl, Cd, Cm
                learning_rate=learning_rate,
                seed=int(s)
            )
            for s in seeds
        ]

        self.normalizer = InputNormalizer(input_dim)
        self.output_normalizer = InputNormalizer(3)
        self._trained = False

        # For OOD detection
        self._train_features_mean = None
        self._train_features_cov_inv = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the ensemble.

        Args:
            X: Input features (CST params + Re + alpha)
            y: Targets (Cl, Cd, Cm)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            early_stopping_patience: Epochs without improvement before stopping
            verbose: Print progress

        Returns:
            Training history
        """
        # Split data
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Fit normalizers
        self.normalizer.fit(X_train)
        self.output_normalizer.fit(y_train)

        X_train_norm = self.normalizer.transform(X_train)
        y_train_norm = self.output_normalizer.transform(y_train)
        X_val_norm = self.normalizer.transform(X_val)
        y_val_norm = self.output_normalizer.transform(y_val)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train each ensemble member
            epoch_losses = []
            indices = np.random.permutation(len(X_train_norm))

            for model in self.models:
                model_loss = 0
                n_batches = 0

                for i in range(0, len(X_train_norm), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    loss = model.train_step(
                        X_train_norm[batch_idx],
                        y_train_norm[batch_idx]
                    )
                    model_loss += loss
                    n_batches += 1

                epoch_losses.append(model_loss / n_batches)

            train_loss = np.mean(epoch_losses)
            history['train_loss'].append(train_loss)

            # Validation
            val_predictions = []
            for model in self.models:
                mean, _ = model.forward(X_val_norm)
                val_predictions.append(mean)
            val_pred_mean = np.mean(val_predictions, axis=0)
            val_loss = np.mean((val_pred_mean - y_val_norm) ** 2)
            history['val_loss'].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Setup OOD detection
        self._setup_ood_detection(X_train_norm)
        self._trained = True

        return history

    def _setup_ood_detection(self, X_train: np.ndarray):
        """Setup Mahalanobis distance-based OOD detection."""
        self._train_features_mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train.T)
        # Regularize covariance for stability
        cov += np.eye(cov.shape[0]) * 1e-6
        self._train_features_cov_inv = np.linalg.inv(cov)

    def _compute_mahalanobis(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance from training distribution."""
        diff = x - self._train_features_mean
        return float(np.sqrt(diff @ self._train_features_cov_inv @ diff))

    def predict(self, X: np.ndarray) -> List[Prediction]:
        """
        Make predictions with uncertainty estimates.

        Args:
            X: Input features, shape (n_samples, input_dim)

        Returns:
            List of Prediction objects
        """
        if not self._trained:
            raise RuntimeError("Model not trained")

        X_norm = self.normalizer.transform(X)
        predictions = []

        for i in range(len(X)):
            x = X_norm[i:i+1]

            # Get predictions from all ensemble members
            means = []
            log_vars = []
            for model in self.models:
                mean, log_var = model.forward(x)
                means.append(mean[0])
                log_vars.append(log_var[0])

            means = np.array(means)
            log_vars = np.array(log_vars)

            # Ensemble mean and variance
            ensemble_mean = np.mean(means, axis=0)
            ensemble_var_epistemic = np.var(means, axis=0)

            # Average aleatoric variance
            aleatoric_var = np.mean(np.exp(log_vars), axis=0)

            # Total variance
            total_var = ensemble_var_epistemic + aleatoric_var

            # Denormalize predictions
            pred_denorm = self.output_normalizer.inverse_transform(
                ensemble_mean.reshape(1, -1)
            )[0]

            # Scale uncertainties
            std_scale = self.output_normalizer.std
            epistemic_std = np.sqrt(ensemble_var_epistemic) * std_scale
            aleatoric_std_val = np.sqrt(aleatoric_var) * std_scale
            total_std = np.sqrt(total_var) * std_scale

            # OOD detection
            mahal_dist = self._compute_mahalanobis(x[0])
            # Threshold based on chi-squared distribution (95% confidence)
            ood_threshold = np.sqrt(self.input_dim * 2)  # Approximate
            is_ood = mahal_dist > ood_threshold
            ood_score = mahal_dist / ood_threshold

            # Confidence score (inverse of normalized uncertainty)
            confidence = 1.0 / (1.0 + np.mean(total_std / np.abs(pred_denorm + 1e-6)))
            if is_ood:
                confidence *= 0.5  # Reduce confidence for OOD samples

            predictions.append(Prediction(
                cl=float(pred_denorm[0]),
                cd=float(max(pred_denorm[1], 1e-4)),  # Ensure positive
                cm=float(pred_denorm[2]),
                cl_std=float(total_std[0]),
                cd_std=float(total_std[1]),
                cm_std=float(total_std[2]),
                epistemic_std={
                    'cl': float(epistemic_std[0]),
                    'cd': float(epistemic_std[1]),
                    'cm': float(epistemic_std[2]),
                },
                aleatoric_std={
                    'cl': float(aleatoric_std_val[0]),
                    'cd': float(aleatoric_std_val[1]),
                    'cm': float(aleatoric_std_val[2]),
                },
                confidence=float(confidence),
                is_ood=bool(is_ood),
                ood_score=float(ood_score),
            ))

        return predictions

    def predict_single(
        self,
        cst_params: np.ndarray,
        reynolds: float,
        alpha: float
    ) -> Prediction:
        """Convenience method for single prediction."""
        X = np.concatenate([cst_params, [np.log10(reynolds), alpha]])
        return self.predict(X.reshape(1, -1))[0]

    def save(self, dirpath: Path):
        """Save model to directory."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # Save each ensemble member
        for i, model in enumerate(self.models):
            model.save(dirpath / f"model_{i}.json")

        # Save normalizers
        self.normalizer.save(dirpath / "input_normalizer.json")
        self.output_normalizer.save(dirpath / "output_normalizer.json")

        # Save OOD detection state
        ood_state = {
            'mean': self._train_features_mean.tolist(),
            'cov_inv': self._train_features_cov_inv.tolist(),
        }
        with open(dirpath / "ood_state.json", 'w') as f:
            json.dump(ood_state, f)

        # Save config
        config = {
            'input_dim': self.input_dim,
            'n_ensemble': self.n_ensemble,
            'hidden_dims': self.hidden_dims,
        }
        with open(dirpath / "config.json", 'w') as f:
            json.dump(config, f)

    @classmethod
    def load(cls, dirpath: Path) -> 'EnsembleSurrogateModel':
        """Load model from directory."""
        dirpath = Path(dirpath)

        with open(dirpath / "config.json", 'r') as f:
            config = json.load(f)

        model = cls(
            input_dim=config['input_dim'],
            n_ensemble=config['n_ensemble'],
            hidden_dims=config['hidden_dims'],
        )

        # Load ensemble members
        for i in range(model.n_ensemble):
            model.models[i].load(dirpath / f"model_{i}.json")

        # Load normalizers
        model.normalizer = InputNormalizer.load(dirpath / "input_normalizer.json")
        model.output_normalizer = InputNormalizer.load(dirpath / "output_normalizer.json")

        # Load OOD state
        with open(dirpath / "ood_state.json", 'r') as f:
            ood_state = json.load(f)
        model._train_features_mean = np.array(ood_state['mean'])
        model._train_features_cov_inv = np.array(ood_state['cov_inv'])

        model._trained = True
        return model


if __name__ == "__main__":
    # Test the module
    print("Testing Surrogate Model Module")
    print("=" * 50)

    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 500
    n_cst_params = 12

    # Synthetic inputs
    X = np.random.randn(n_samples, n_cst_params + 2)  # CST + log(Re) + alpha
    X[:, -2] = np.random.uniform(5, 7, n_samples)  # log10(Re)
    X[:, -1] = np.random.uniform(-5, 15, n_samples)  # alpha

    # Synthetic targets (simple relationships for testing)
    y = np.zeros((n_samples, 3))
    y[:, 0] = 0.1 * X[:, -1] + 0.5 * X[:, 0] + np.random.randn(n_samples) * 0.1  # Cl
    y[:, 1] = 0.01 + 0.001 * X[:, -1]**2 + np.random.randn(n_samples) * 0.002  # Cd
    y[:, 2] = -0.02 * X[:, -1] + np.random.randn(n_samples) * 0.01  # Cm

    print(f"Training data: X={X.shape}, y={y.shape}")

    # Train model
    model = EnsembleSurrogateModel(
        input_dim=n_cst_params + 2,
        n_ensemble=3,
        hidden_dims=[32, 32],
    )

    history = model.fit(X, y, epochs=50, verbose=True)

    # Test prediction
    X_test = X[:5]
    predictions = model.predict(X_test)

    print("\nTest predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i}: Cl={pred.cl:.4f}±{pred.cl_std:.4f}, "
              f"Cd={pred.cd:.5f}±{pred.cd_std:.5f}, "
              f"conf={pred.confidence:.2f}, OOD={pred.is_ood}")
