def price_bond_option(
    option_type: Literal["call", "put"],
    P0_T1: float,
    P0_T2: float,
    strike: float,
    volatility: float,
    T1: float,
) -> float:
    """
    Prices a European option on a zero-coupon bond using Black's model.

    Parameters:
        option_type (Literal["call", "put"]): Type of option ('call' or 'put').
        P0_T1 (float): Zero-coupon bond price for maturity T1 (option expiry).
        P0_T2 (float): Zero-coupon bond price for maturity T2 (bond underlying).
        strike (float): Strike price of the option.
        volatility (float): Volatility of the forward bond price.
        T1 (float): Time to option expiry (in years).

    Returns:
        float: The price of the bond option.

    Raises:
        ValueError: If an invalid option_type is provided.
    """
    if T1 <= 0:
        raise ValueError("Option expiry T1 must be positive.")
    if volatility <= 0:
        raise ValueError("Volatility must be positive.")

    # Calculate d1 and d2 as per Black's formula
    # The forward price for the bond is F = P0_T2 / P0_T1.
    F = P0_T2 / P0_T1
    sigma_sqrt_T1 = volatility * sqrt(T1)
    d1 = (log(F / strike) + 0.5 * volatility**2 * T1) / sigma_sqrt_T1
    d2 = d1 - sigma_sqrt_T1

    # Black's formula for a European call option on a zero-coupon bond:
    call_price = P0_T2 * norm.cdf(d1) - strike * P0_T1 * norm.cdf(d2)
    logging.debug("Computed d1=%.4f, d2=%.4f, call_price=%.6f", d1, d2, call_price)

    if option_type.lower() == "call":
        price = call_price
    elif option_type.lower() == "put":
        # Using put-call parity:
        # Put price = Call price - P0_T2 + strike * P0_T1
        price = call_price - P0_T2 + strike * P0_T1
    else:
        raise ValueError("Invalid option_type. Must be 'call' or 'put'.")

    logging.info("%s option price computed: %.6f", option_type.capitalize(), price)
    return price
