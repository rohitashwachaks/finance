import pandas as pd
import numpy as np
import gurobipy as gb
import yfinance as yf

class TradingAlgo():  
    def __init__(self, name: str) -> None:
        self.identifier = name
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        pass


class SNP500_Algo(TradingAlgo):    
    def __init__(self) -> None:
        super().__init__("MarketCap Weighted Portfolio")
        pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
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
    
    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        weights = pd.Series([0.3,0.25,0.2,0.15, 0.1], index = price.index)*investment#.apply(math.floor)
        weights = weights.divide(price["close"])
        return weights

class capm_Algo(TradingAlgo):
    def __init__(self, short: bool = False) -> None:
        self.isShortAllowed = short
        super().__init__("CAPM Weights")
        pass

    def calculate_inputs(self, date: pd.Timestamp)->None:
        # stocks = yf.Tickers(" ".join(self.tics))
        stock_data = yf.download(" ".join(self.tics), start=(date -  pd.Timedelta("5 Y")), end=date)["Close"].dropna()
        stock_data = stock_data.pct_change()[1:]
        
        self.sigma = np.cov(stock_data.cov())
        # print("Sigma Matrix Calculated. Shape:",self.sigma.shape)

        self.expected_returns = np.array(stock_data.mean())
        # print("Expected Returns are:",self.expected_returns)

        return

    def maximise_sharpe(self):
        variables = len(self.tics)

        constraints = 1     # Shorting of stocks allowed
        if not self.isShortAllowed:
            constraints += variables      # No-Shorting of stocks allowed

        # Defining Model's Decision Variables
        maxSharpe_model = gb.Model("MaxSharpe")
        model_X = maxSharpe_model.addMVar(variables)
        
        # Defining Model's Objective Function (Minimize Risk)
        risk = model_X@ self.sigma @model_X
        maxSharpe_model.setObjective(risk, sense=gb.GRB.MINIMIZE)

        # Defining Constraints
        A = np.zeros((constraints,variables)) # initialize constraint matrix
        A[0] = self.expected_returns # Sum of weights = 1
        if not self.isShortAllowed:
            A[1:] = np.diag(self.expected_returns) # No shorting

        b = np.array([1]+[0]*(constraints-1))

        sense = np.array(["="]+[">"]*(constraints-1))

        model_constraints = maxSharpe_model.addMConstrs(A, model_X, sense, b)
        # print_equations(np.array([0]*variables), A, sense,b)

        # Optimize Model
        maxSharpe_model.Params.OutputFlag = 0
        maxSharpe_model.optimize()

        optimal_obj = maxSharpe_model.objVal
        optimal_values = model_X.x

        efficient_weights = np.round(np.multiply(self.expected_returns,optimal_values),2)
        return efficient_weights
        # efficient_returns = efficient_weights@e_returns.T
        # efficient_vol = np.sqrt(efficient_weights.T@covarience@efficient_weights)

        # print("Efficent Portfolio Weights:", efficient_weights)
        # print("Efficient Portfolio E(returns): ",efficient_returns*100,"%")
        # print("Efficient Portfolio Volatility: ",np.round(efficient_vol,5))
        # pass

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        self.tics = price.index.tolist()
        self.calculate_inputs(date)
        weights = pd.Series(self.maximise_sharpe(), index = price.index)*investment#.apply(math.floor)
        weights = weights.divide(price["close"])
        return weights