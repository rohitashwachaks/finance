import pandas as pd
import numpy as np
import math

class TradingAlgo():  
    def __init__(self, name: str) -> None:
        self.identifier = name
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.DateTime) -> pd.Series(dtype=float):
        pass


class SNP500_Algo(TradingAlgo):    
    def __init__(self) -> None:
        super().__init__("MarketCap Weighted Portfolio")
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.DateTime) -> pd.Series(dtype=float):
        price["cap"] = price["shares"]*price["close"]
        nav = sum(price["cap"])

        weights = (((price["cap"]/nav)*investment)/price["close"])#.apply(math.floor)
        return weights


class DJIA_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("Price Cap Weighted")
        pass

    def run(self, price: pd.DataFrame,
                investment: float) -> pd.Series(dtype=float):
        total_price = sum(price["close"])

        weights = [investment/total_price]*len(price.index)
        weights = pd.Series(weights, index= price.index)#.apply(math.floor)
        return weights


class constant_weight_Algo(TradingAlgo):    
    def __init__(self) -> None:
        super().__init__("Asset Class Weighted (Constant)")
        pass
    
    def run(self, price: pd.DataFrame, investment: float, date: pd.DateTime) -> pd.Series(dtype=float):
        weights = pd.Series([0.3,0.25,0.2,0.15, 0.1], index = price.index)*investment#.apply(math.floor)
        weights = weights.divide(price["close"])
        return weights

class capm_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("CAPM Weights")
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.DateTime) -> pd.Series(dtype=float):
        tics = price.index.tolist()
        print(tics, date)
        weights = pd.Series([0,0,0.81,0.19,0], index = price.index)*investment#.apply(math.floor)
        weights = weights.divide(price["close"])
        return weights
