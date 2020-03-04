import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


class ModelEvaluation:
    plt.style.use("ggplot")
    
    def __init__(self,
                test_data_param: dict):
        
        self.test_data_param = test_data_param
        self.models = list(test_data_param.keys())
        self.metrics = {}
        self.df_group_list = {}
        self.df_list ={}
        for model in self.test_data_param.keys():
            pred = self.test_data_param[model]["pred_col"]
            act = self.test_data_param[model]["actual"]
            self.df_list[model] = pd.read_csv(self.test_data_param[model]["loc"])
            self.metrics[model]= {"r2": r2_score(self.df_list[model][act], self.df_list[model][pred]),
                                  "MAE": mean_absolute_error(self.df_list[model][act], self.df_list[model][pred])
                                 }
            
        self.metrics = pd.DataFrame(self.metrics)
        
        
    def plot_scatter(self, models=None, x_lims=None, y_lims=None):
        if models is None:
            models = self.models
        else:
            models = models
                
        plt.figure(figsize=(8,8))
        
        for model in models:
            
            actual = self.test_data_param[model]["actual"]
            pred = self.test_data_param[model]["pred_col"]
            col = self.test_data_param[model]["col"]
            
            plt.scatter(self.df_list[model][actual],
                        self.df_list[model][pred],
                        color=col, alpha=0.5,
                        label=f"{model} predicted depth")
            
        plt.plot(self.df_list[model][actual],
                    self.df_list[model][actual],
                    color="black",
                    linestyle="--")
        plt.legend()
        plt.title("Actual vs Predicted depth (ft)")
        if x_lims is not None:
            x_low, x_high = x_lims
            plt.xlim(left=x_low, right=x_high)
        if y_lims is not None:
            y_low, y_high = y_lims
            plt.ylim(bottom=y_low, top=y_high)
        plt.xlabel("Actual depth (ft)")
        plt.xlabel("predicted depth (ft)")
        plt.show()
    
    def plot_error_dist(self, bins, models=None, x_lims=None, y_lims=None):
        if models is None:
            models = self.models
        else:
            models = models
                
        plt.figure(figsize=(8,8))
        
        for model in models:
            
            actual = self.test_data_param[model]["actual"]
            pred = self.test_data_param[model]["pred_col"]
            col = self.test_data_param[model]["col"]
            
            plt.hist(self.df_list[model][actual] - self.df_list[model][pred],
                    color=col,
                    bins=bins,
                    alpha=0.5,
                    ec="black",
                    label=f"{model} predicted depth")
            
        plt.legend()
        plt.title("Actual vs Predicted depth (ft)")
        if x_lims is not None:
            x_low, x_high = x_lims
            plt.xlim(left=x_low, right=x_high)
        if y_lims is not None:
            y_low, y_high = y_lims
            plt.ylim(bottom=y_low, top=y_high)
        plt.xlabel("Actual depth (ft)")
        plt.xlabel("predicted depth (ft)")
        plt.show()
        
    def _calc_metric(self, df, metric, pred, actual):
        try:
            if metric =="r2":
                df[metric] = r2_score(df[actual], df[pred])
            elif metric == "MAE":
                df[metric] = mean_absolute_error(df[actual], df[pred])
            else:
                raise ValueError("Metric not supported")
            return df.iloc[0]
        except ValueError:
            pass
        
    
    def plot_metric_vs_depth(self, min_value, max_value, step, metric="MAE", models=None,):
        if models is None:
            models = self.models
        else:
            models = models
        
        plt.figure(figsize=(12,8))
        for model in models:
            actual = self.test_data_param[model]["actual"]
            pred = self.test_data_param[model]["pred_col"]
            col = self.test_data_param[model]["col"]
            bins = pd.cut(self.df_list[model][actual], bins=np.arange(min_value, max_value, step))
            self.df_list[model]["bins"] = bins
            self.bins = bins
            self.df_group_list[f"{model}_GROUPED_DEPTH"] = self.df_list[model].groupby("bins").apply(lambda x: self._calc_metric(x,metric,pred,actual))
            self.df_group_list[f"{model}_GROUPED_DEPTH"].index = (self.df_group_list[f"{model}_GROUPED_DEPTH"].index.left + self.df_group_list[f"{model}_GROUPED_DEPTH"].index.right) /2
            self.df_group_list[f"{model}_GROUPED_DEPTH"].dropna(inplace=True)
            plt.plot(self.df_group_list[f"{model}_GROUPED_DEPTH"].index,
                     self.df_group_list[f"{model}_GROUPED_DEPTH"][metric],
                     color=col,
                    label= f"{model} {metric} by depth")
        plt.legend()
        plt.title(f"Model {metric} grouped by depth (bins=({min_value}, {max_value}), step={step})")
        plt.xlabel("Depth (ft)")
        plt.ylabel(f"{metric}")
        plt.show()
        plt.show()
            
            
            
            
            
    def plot_metric_vs_surface(self, metric="MAE", models=None):
        if models is None:
            models = self.models
        else:
            models = models
        
        plt.figure(figsize=(12, 8))
        for model in models:
            actual = self.test_data_param[model]["actual"]
            pred = self.test_data_param[model]["pred_col"]
            col = self.test_data_param[model]["col"]
            
            self.df_group_list[f"{model}_GROUPED_SURFACE"] = self.df_list[model].groupby("Surface").apply(lambda x: self._calc_metric(x,metric,pred,actual))
            self.df_group_list[f"{model}_GROUPED_SURFACE"].dropna(inplace=True)
            plt.plot(self.df_group_list[f"{model}_GROUPED_SURFACE"].index,
                     self.df_group_list[f"{model}_GROUPED_SURFACE"][metric],
                     color=col,
                     label=f"{model} {metric} by surface")
        plt.legend()
        plt.title(f"Model {metric} by grouped by surface")
        plt.xticks(rotation=90)
        plt.xlabel("surface name")
        plt.ylabel(f"{metric}")
        plt.show()
            
            
        