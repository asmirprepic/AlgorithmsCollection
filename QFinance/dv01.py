def dv01(self, valuation_date: datetime, yield_curve: YieldCurve) -> float:
        """
        Calculates the DV01 (Dollar Value of a 01) or PV01 (Present Value of a Basis Point) of the bond.

        DV01 estimates the change in bond value for a 1 basis point (0.01%) increase in yields.

        Args:
            valuation_date (datetime): Date for DV01 calculation.
            yield_curve (YieldCurve): Baseline yield curve.

        Returns:
            float: DV01 value (positive value, representing value decrease for yield increase).
        """
        initial_pv = self.present_value(valuation_date, yield_curve)

        bumped_rates = yield_curve.rates + 0.0001 # Add 1 bps to all rates
        bumped_yc_data = {'Date': yield_curve.dates, 'Rate': bumped_rates}
        bumped_yield_curve = YieldCurve.from_dataframe(pd.DataFrame(bumped_yc_data), interpolation_method=yield_curve.interpolation_method)

        bumped_pv = self.present_value(valuation_date, bumped_yield_curve)

        dv01_val = initial_pv - bumped_pv
        return dv01_val
