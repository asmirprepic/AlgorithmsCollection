### Pricing bond and plotting


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bond_price(coupon_rate, face_value, time_to_maturity, yield_to_maturity):
    """
    Calculate the price of a bond.

    Parameters:
    coupon_rate (float): The annual coupon rate of the bond.
    face_value (float): The face value of the bond.
    time_to_maturity (int): The time to maturity of the bond in years.
    yield_to_maturity (float): The yield to maturity of the bond.

    Returns:
    float: The present value (price) of the bond.
    """


    coupon_payment = coupon_rate * face_value # Annual coupon payment
    discount_factor = 1 / (1 + yield_to_maturity) ** time_to_maturity # Discount factor for the bond
    present_value = 0 # Initialize present value

    # Calculate present value of each cash flow
    for t in range(1, time_to_maturity + 1):
        cash_flow = coupon_payment if t < time_to_maturity else coupon_payment + face_value
        present_value += cash_flow * discount_factor ** t
    return present_value

# Example usage
coupon_rate = 0.05
face_value = 1000
time_to_maturity = 5
yield_to_maturity = 0.06

price = bond_price(coupon_rate, face_value, time_to_maturity, yield_to_maturity)
print("The price of the bond is:", round(price, 2))


### 3D plot of bond Price
yields = np.linspace(0.01, 0.1, 10)  # Range of yields
times = np.arange(1, 21)  # Range of times to maturity

# Calculate bond prices for each combination of yield and time values
bond_prices = np.zeros((len(yields), len(times)))  # Initialize empty array
for i, y in enumerate(yields):
    for j, t in enumerate(times):
        bond_prices[i, j] = bond_price(0.05, 1000, t, y)  # Use bond_price function from previous example

bond_prices = bond_prices.T
# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(yields, times)
ax.plot_surface(X, Y, bond_prices, cmap='viridis')
ax.set_xlabel('Yield to Maturity')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Bond Price')
plt.show()
