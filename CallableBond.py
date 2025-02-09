class CallableBondPricer:
    """
    Class for pricing callable bonds using a recombining binomial interest rate tree.

    The model constructs a tree for the short rate starting from an initial rate (r0)
    and applying up and down moves each period according to a specified volatility.
    The callable bond is defined by its face value, periodic coupon, maturity (in discrete periods),
    and a call schedule. At the call dates, the bond's value is capped by the call price.

    The binomial tree is built as follows:
        - Up factor: u = exp(sigma * sqrt(dt))
        - Down factor: d = 1/u
        - The risk-neutral probability is set to p (default 0.5 for a symmetric tree).
    """

    def __init__(
        self,
        r0: float,
        sigma: float,
        dt: float,
        maturity: int,
        coupon: float,
        face_value: float,
        call_schedule: Optional[Dict[int, float]] = None,
        p: float = 0.5,
    ) -> None:
        """
        Initializes the CallableBondPricer.

        Parameters:
            r0 (float): Initial short rate.
            sigma (float): Volatility of the short rate.
            dt (float): Time step in years (assumed to coincide with coupon dates).
            maturity (int): Maturity in number of periods.
            coupon (float): Coupon payment per period.
            face_value (float): Face (par) value of the bond.
            call_schedule (Optional[Dict[int, float]]): A dictionary mapping period indices (int)
                to call prices. At these periods, the bond's value is capped by the call price.
            p (float): Risk-neutral probability for an up move (default is 0.5).
        """
        self.r0 = r0
        self.sigma = sigma
        self.dt = dt
        self.maturity = maturity
        self.coupon = coupon
        self.face_value = face_value
        self.call_schedule = call_schedule if call_schedule is not None else {}
        self.p = p

        # Compute up and down factors for the binomial tree.
        self.u = math.exp(sigma * math.sqrt(dt))
        self.d = 1.0 / self.u

        logging.info("Initialized CallableBondPricer with r0=%.4f, sigma=%.4f, dt=%.4f, maturity=%d, coupon=%.4f, face_value=%.4f",
                     r0, sigma, dt, maturity, coupon, face_value)

    def build_rate_tree(self) -> List[List[float]]:
        """
        Builds a recombining binomial tree for short rates.

        Returns:
            List[List[float]]: A list of lists where each inner list contains the short rates
            at that discrete time period.
        """
        tree: List[List[float]] = []
        for i in range(self.maturity + 1):
            level_rates = []
            for j in range(i + 1):
                # For node (i, j): j up moves and (i - j) down moves.
                rate = self.r0 * (self.u ** j) * (self.d ** (i - j))
                level_rates.append(rate)
            tree.append(level_rates)
            logging.debug("Rate tree level %d: %s", i, level_rates)
        return tree

    
