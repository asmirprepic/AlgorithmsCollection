import numpy as np
from typing import Tuple

def cds_price(notional: price, recovery_rate: float, credit_spred: float, discount_rate: float, maturity: int, payment_freq: int = 4) -> Tuple[float,float,float]:

  """
  Calculate the price of a Credit Default Swap (CDS) under a constant hazard rate. 

  Returns: 
  ------------
  - Tuple[float,float,float]: A tuple containing:
    - cds_price (float): The fair value of the cds price
    - premium_leg_value (float): Present value of premium leg (spread payments)
    - protection_leg_value (float): Present value of the protection leg (expected default)
  """

  if not (0 <= recovery_rate <= 1):
        raise ValueError("Recovery rate must be between 0 and 1.")
  if credit_spread < 0:
      raise ValueError("Credit spread must be non-negative.")
  if discount_rate < 0:
    raise ValueError("Discount rate must be non-negative.")
  if maturity <= 0:
    raise ValueError("Maturity must be positive.")
  if payment_freq <= 0:
    raise ValueError("Payment frequency must be positive.")

  # Derived constant hazard rate
  hazard_rate = credit_spread/(1-recovery_rate)
  time_steps = np.arange(1, maturity*payment_freq +1)/payment_freq

  discount_factors = np.exp(-discount_rate*time_steps)
  survival_probabilities = np.exp(-hazard_rate*time_steps)
  premium_cash_flows = notional * (credit_spread / payment_freq) * discount_factors * survival_probabilities
  premium_leg_value = np.sum(premium_cash_flows)

  # Protection leg: Expected payout by the protection seller in case of default
  protection_cash_flows = notional * (1 - recovery_rate) * (1 - survival_probabilities) * discount_factors
  protection_leg_value = np.sum(protection_cash_flows)

  # CDS Price = Protection Leg Value - Premium Leg Value
  cds_price = protection_leg_value - premium_leg_value

  return cds_price, premium_leg_value, protection_leg_value
