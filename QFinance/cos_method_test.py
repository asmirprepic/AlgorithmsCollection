def cos_method_option_price(cf, S0, K, T, r, N, a, b, option_type='call'):
 
    
    def chi(k, a, b, c1, c2):
        if k == 0:
            return b - a
        else:
            return (np.sin(k * np.pi * (b - a) / (c2 - c1)) - np.sin(k * np.pi * (a - a) / (c2 - c1))) / (k * np.pi)

    def psi(k, a, b, c1, c2):
        return np.exp(-k**2 * np.pi**2 * T / (c2 - c1))

    
    k = np.arange(N)
    cos_k = np.cos(k * np.pi * (K - a) / (b - a))
    
    
    cf_values = np.array([cf(k * np.pi / (b - a)) for k in range(N)])
    
    
    price = np.exp(-r * T) * (cf_values @ cos_k)
    
    if option_type == 'put':
        return max(0, price)
    else:
        return price
