import math
import matplotlib.pyplot as plt
class BootStrapYC:
    
    def __init__(self):
        self.zero_rates = dict()
        self.instruments = dict()
        
        
    def add_instrument(self,par,T,coup,price,compounding_freq=2):
        self.instruments[T] = (par,coup,price,compounding_freq)
    
    
    def get_maturities(self):
        return sorted(self.instruments.keys())
    
    def get_zero_rates(self):
        self.bootstrap_zero_coupons()
        self.get_bond_spot_rates()
        
        return [self.zero_rates[T] for T in self.get_maturities()]
    
    def bootstrap_zero_coupons(self):
        
        for (T,instrument) in self.instruments.items():
            (par,coup,price,freq) = instrument
            
            if coup == 0:
                spot_rate  =self.zero_coupon_spot_rate(par,price,T)
                self.zero_rates[T] = spot_rate
    
    def get_bond_spot_rates(self):
        
        for T in self.get_maturities():
            instrument = self.instruments[T]
            (par,coup,price,freq) = instrument
            
            if coup != 0:
                spot_rate=self.calculate_bond_spot_rate(T,instrument)
                self.zero_rates[T] = spot_rate
    
    def zero_coupon_spot_rate(self,par,price,T):
        spot_rate = math.log(par/price)/T
        return spot_rate 
    
    def calculate_bond_spot_rate(self,T,instrument):
        try:
            (par,coup,price,freq) = instrument
            periods = T*freq
            value = price
            per_coupon = coup/freq
            for i in range(int(periods)-1):
                t = (i+1)/float(freq)
                spot_rate=self.zero_rates[t]
                discounted_coupon = per_coupon*math.exp(-spot_rate*t)
                value -= discounted_coupon
            last_period=int(periods)/float(freq)
            spot_rate=-math.log(value/(par+per_coupon))/last_period
            return spot_rate
        except Exception as e:
            print("Spot rate not found for T=","Error: ",str(e))

if __name__=="__main__":
    yield_curve = BootStrapYC()
    yield_curve.add_instrument(100, 0.25, 0., 97.5)
    yield_curve.add_instrument(100, 0.5, 0., 94.9)
    yield_curve.add_instrument(100, 1.0, 0., 90.)
    yield_curve.add_instrument(100, 1.5, 8, 96., 2)
    yield_curve.add_instrument(100, 2., 12, 101.6, 2)
    
    y = yield_curve.get_zero_rates()
    x = yield_curve.get_maturities()
    
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, y)
    plt.title("Zero Curve") 
    plt.ylabel("Zero Rate (%)")
    plt.xlabel("Maturity in Years");
