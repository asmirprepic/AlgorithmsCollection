class ForwardRates:
    def __init__(self):
        self.forward_rates = []
        self.spot_rates = dict()
        
    def add_spot_rate(self,T,spot_rate):
        self.spot_rates[T] = spot_rate
        
    def get_forward_rates(self):
        periods = sorted(self.spot_rates.keys())
        for T2,T1 in zip(periods,periods[1:]):
            forward_rate = self.calculate_forward_rate(T1,T2)
            self.forward_rates.append(forward_rate)
        return self.forward_rates
    
    def calculate_forward_rate(self,T1,T2):
        R1 = self.spot_rates[T1]
        R2= self.spot_rates[T2]
        forward_rate = (R2*T2-R1*T1)/(T2-T1)
        return forward_rate
    
if __name__ == "__main__":
    fr = ForwardRates()
    fr.add_spot_rate(0.25, 10.127)
    fr.add_spot_rate(0.50, 10.469)
    fr.add_spot_rate(1.00, 10.536)
    fr.add_spot_rate(1.50, 10.681)
    fr.add_spot_rate(2.00, 10.808)
    print(fr.get_forward_rates())
