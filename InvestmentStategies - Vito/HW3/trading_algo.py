import pandas as pd
import numpy as np
import math

class TradingAlgo():
    def __init__(self, name: str) -> None:
        self.identifier = name
        pass

    def run(self, price: pd.DataFrame, investment: float) -> pd.Series(dtype=float):
        pass

class SNP500_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__(self, "MarketCap Weighted Portfolio")
        self.identifier = super().identifier
        return

    def run(self, price: pd.DataFrame,
                investment: float) -> pd.Series(dtype=float):
        price["cap"] = price["shares"]*price["close"]
        nav = sum(price["cap"])

        weights = (((price["cap"]/nav)*investment)/price["close"])#.apply(math.floor)
        return weights

class DJIA_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__(self, "Price Cap Weighted")
        self.identifier = super().identifier
        return

    def run(self, price: pd.DataFrame,
                investment: float) -> pd.Series(dtype=float):
        total_price = sum(price["close"])

        weights = [investment/total_price]*len(price.index)
        weights = pd.Series(weights, index= price.index)#.apply(math.floor)
        return weights

class constant_weight_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__(self, "Asset Class Weighted (Constant)")
        self.identifier = super().identifier
        return
    
    def run(self, price: pd.DataFrame, investment: float) -> pd.Series(dtype=float):
        nav = np.dot(price["shares"], price["close"])
        weights = pd.Series([0.3,0.25,0.2,0.15, 0.1], index = price.index)*nav#.apply(math.floor)
        return weights
