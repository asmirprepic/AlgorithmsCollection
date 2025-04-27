def calculate_liquidity_factor(data) -> pd.DataFrame:
  """
  Calculate the liquidity factor based on normalized spreads and market depth.
  """
  try:
      data['norm_spread'] = (data['spread'] - data['spread'].mean()) / data['spread'].std()
      data['norm_depth'] = (data['depth'] - data['depth'].mean()) / data['depth'].std()
      data['liquidity_factor'] = 0.7 * data['norm_spread'] + 0.3 * (1 / (1 + data['norm_depth']))
      logging.info("Liquidity factor calculated successfully.")
      return data
