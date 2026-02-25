"""
Centralized configuration management for AUTOPSY Trading Engine.
Implements validation, type safety, and Firebase integration.
"""
import os
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """Exchange API configuration"""
    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")
    exchange_name: str = os.getenv("EXCHANGE_NAME", "binance")
    use_sandbox: bool = os.getenv("USE_SANDBOX", "True").lower() == "true"
    rate_limit: int = int(os.getenv("RATE_LIMIT", "100"))
    
    def __post_init__(self):
        """Validate exchange configuration"""
        if not self.api_key and not self.use_sandbox:
            logger.warning("No API key provided but sandbox mode is disabled")
        if self.rate_limit < 1 or self.rate_limit > 1000:
            raise ValueError(f"Invalid rate limit: {self.rate_limit}")

@dataclass
class QuantumConfig:
    """Quantum-inspired algorithm configuration"""
    use_quantum_tunneling: bool = True
    temperature: float = 1.0
    cooling_rate: float = 0.95
    num_qubits: int = 10
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    
    def __post_init__(self):
        """Validate quantum parameters"""
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive: {self.temperature}")
        if not 0 < self.cooling_rate < 1:
            raise ValueError(f"Cooling rate must be between 0 and 1: {self.cooling_rate}")

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
    stop_loss_percentage: float = float(os.getenv("STOP_LOSS", "0.02"))
    take_profit_percentage: float = float(os.getenv("TAKE_PROFIT", "0.03"))
    max_leverage: int = int(os.getenv("MAX_LEVERAGE", "3"))
    
    def __post_init__(self):
        """Validate risk parameters"""
        if not 0 < self.max_position_size <= 1:
            raise ValueError(f"Invalid max position size: {self.max_position_size}")
        if not 0 < self.max_daily_loss <= 1:
            raise ValueError(f"Invalid max daily loss: {self.max_daily_loss}")
        if self.max_leverage < 1 or self.max_leverage > 100:
            raise ValueError(f"Invalid leverage: {self.max_leverage}")

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.001"))
    discount_factor: float = float(os.getenv("DISCOUNT_FACTOR", "0.95"))
    epsilon: float = float(os.getenv("EPSILON", "1.0"))
    epsilon_decay: float = float(os.getenv("EPSILON_DECAY", "0.995"))
    min_epsilon: float = float(os.getenv("MIN_EPSILON", "0.01"))
    batch_size: int = int(os.getenv