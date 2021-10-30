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

class retrospective_sharpe_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("Retrospective Max Weights")
        return

    def init_params(self, ticker_list: list,
                    lever: float = 0.0,
                    max_cap: float = 0.45,
                    start_date: pd.Timestamp = pd.to_datetime("12/31/2010"),
                    end_date: pd.Timestamp = pd.to_datetime("12/31/2020")) -> None:
        self.tics = ticker_list
        self.max_leverage = lever
        self.max_allocation = max_cap

        self.calculate_inputs(start_date, end_date)
        self.maximise_sharpe()
        return

    def calculate_inputs(self, start_date: pd.Timestamp, end_date: pd.Timestamp = pd.to_datetime("12/31/2020"))->None:
        stock_data = yf.download(" ".join(self.tics), start=start_date, end=end_date)["Close"].dropna()
        stock_data = stock_data.pct_change()[1:]
        
        self.sigma = np.cov(stock_data.cov())
        self.expected_returns = np.array(stock_data.mean())
        return

    def maximise_sharpe(self):
        variables = len(self.tics)

        # Defining Model's Decision Variables
        maxSharpe_model = gb.Model("MaxSharpe")
        y = maxSharpe_model.addMVar(variables, 
                                        lb = -self.max_leverage,
                                        ub = self.max_allocation)
        
        # Defining Model's Objective Function (Minimize Risk)
        risk = y @ self.sigma @ y
        maxSharpe_model.setObjective(risk, sense=gb.GRB.MINIMIZE)

        # Defining Constraints
        A = np.ones((1,variables)) # Sum of weights = 1
        b = np.array([1])
        sense = np.array(["="])
        maxSharpe_model.addMConstrs(A, y, sense, b)

        # Optimize Model
        maxSharpe_model.Params.OutputFlag = 0
        maxSharpe_model.optimize()

        # optimal_obj = maxSharpe_model.objVal
        optimal_values = y.x

        self.weights = np.round(optimal_values/optimal_values.sum(), 2)
        return

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        if self.weights is None:
            self.tics = price.index.tolist()
            self.calculate_inputs(date)
            self.maximise_sharpe()
        weights = pd.Series(self.weights, index = price.index)*investment#.apply(math.floor)
        weights = weights.divide(price["close"])
        return weights


class capm_Algo(TradingAlgo):
    def __init__(self) -> None:
        super().__init__("CAPM Weights")
        self.tics = None
        pass

    def init_params(self,
                    lever: float = 0.0,
                    max_cap: float = 0.45) -> None:
        self.max_leverage = lever
        self.max_allocation = max_cap
        return

    def calculate_inputs(self, date: pd.Timestamp)->None:
        stock_data = yf.download(" ".join(self.tics), start=(date -  pd.Timedelta("5 Y")), end=date)["Close"].dropna()
        stock_data = stock_data.pct_change()[1:]
        
        self.sigma = np.cov(stock_data.cov())
        self.expected_returns = np.array(stock_data.mean())
        return

    def maximise_sharpe(self)-> np.array:
        variables = len(self.tics)

        # Defining Model's Decision Variables
        maxSharpe_model = gb.Model("MaxSharpe")
        y = maxSharpe_model.addMVar(variables, 
                                        lb = -self.max_leverage,
                                        ub = self.max_allocation)
        
        # Defining Model's Objective Function (Minimize Risk)
        risk = y @ self.sigma @ y
        maxSharpe_model.setObjective(risk, sense=gb.GRB.MINIMIZE)

        # Defining Constraints
        A = np.ones((1,variables)) # Sum of weights = 1
        b = np.array([1])
        sense = np.array(["="])
        maxSharpe_model.addMConstrs(A, y, sense, b)

        # Optimize Model
        maxSharpe_model.Params.OutputFlag = 0
        maxSharpe_model.optimize()

        # optimal_obj = maxSharpe_model.objVal
        optimal_values = y.x

        weights = np.round(optimal_values/optimal_values.sum(), 2)
        return weights

    def run(self, price: pd.DataFrame, investment: float, date: pd.Timestamp) -> pd.Series(dtype=float):
        if self.tics is None:
            self.tics = price.index.tolist()
        self.calculate_inputs(date)
        weights = pd.Series(self.maximise_sharpe(), index = price.index)*investment#.apply(math.floor)
        weights = weights.divide(price["close"])
        return weights


#------

def print_equations(obj, constraint, sense, b):
    print("Optimise System of equations:")
    for i in range(constraint.shape[0]):
        char = "a"
        print("\t"+str(i+1)+")",end=" ")
        for j in range(constraint.shape[1]):
            print(str(constraint[i,j])+char,end=" + ")
            char = chr(ord(char) + 1)
        print("\b\b "+sense[i]+"= "+str(b[i]))
    char = "a"
    print("Subject to:", end=" ")
    for item in obj:
        print(str(item)+char,end=" + ")
        char = chr(ord(char) + 1)
    print("\b\b")
    return