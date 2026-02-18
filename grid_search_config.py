import torch
import numpy as np

GRID_SEARCH_INFO = {}

#################### ks
KS_INFL = {5: [1 + 0.02*i for i in range(11)],
       10: [1 + 0.02*i for i in range(11)],
       15: [1 + 0.02*i for i in range(11)],
       20: [1 + 0.01*i for i in range(11)],
       40: [1 + 0.01*i for i in range(11)],
       60: [1 + 0.01*i for i in range(6)],
       100: [1 + 0.01*i for i in range(6)],
       }

KS_INFL_LETKF = {5: [1 + 0.02*i for i in range(11)],
       10: [1 + 0.02*i for i in range(11)],
       15: [1 + 0.02*i for i in range(11)],
       20: [1 + 0.01*i for i in range(11)],
       40: [1 + 0.01*i for i in range(11)],
       60: [1 + 0.01*i for i in range(6)],
       100: [1 + 0.01*i for i in range(6)],
       }

KS_LOC = {5: [0.001,] + [i for i in range(1,15,2)],
       10: [0.001,] + [i for i in range(1,15,2)],
       15: [0.001,] + [i for i in range(1,15,2)],
       20: [0.001,] + [i for i in range(1,15,2)],
       40: [0.001,] + [i for i in range(1,15,2)],
       60: [0.001,] + [i for i in range(1,15,2)],
       100: [0.001,] + [i for i in range(1,15,2)],
       }

GRID_SEARCH_INFO["ks"] = {
       "sigma_y":[1, 0.5, 0.1],  
       # "methods":["LETKF", "EnKF_Sqrt", "EnKF_PertObs", "iEnKS"],
       "methods":["EnKF", "ESRF", "LETKF", "iEnKS-PertObs"],
       "N_list":[5,10,15,20,40,60,100],
       "infl_list":KS_INFL,
       "letkf_infl_list":KS_INFL_LETKF,
       "loc_rad_list":KS_LOC,
       "num_trials": 1
       }


# ###################### L96
L96_INFL = {5: [1 + 0.02*i for i in range(11)],
       10: [1 + 0.02*i for i in range(11)],
       15: [1 + 0.02*i for i in range(11)],
       20: [1 + 0.01*i for i in range(11)],
       40: [1 + 0.01*i for i in range(11)],
       60: [1 + 0.01*i for i in range(6)],
       100: [1 + 0.01*i for i in range(6)],
       }

L96_INFL_LETKF = {5: [1 + 0.02*i for i in range(11)],
       10: [1 + 0.02*i for i in range(11)],
       15: [1 + 0.02*i for i in range(11)],
       20: [1 + 0.01*i for i in range(11)],
       40: [1 + 0.01*i for i in range(11)],
       60: [1 + 0.01*i for i in range(6)],
       100: [1 + 0.01*i for i in range(6)],
       }

L96_LOC = {5: [0.001,] + [i for i in range(1,15,2)],
       10: [0.001,] + [i for i in range(1,15,2)],
       15: [0.001,] + [i for i in range(1,15,2)],
       20: [0.001,] + [i for i in range(1,15,2)],
       40: [0.001,] + [i for i in range(1,15,2)],
       60: [0.001,] + [i for i in range(1,15,2)],
       100: [0.001,] + [i for i in range(1,15,2)],
       }

GRID_SEARCH_INFO["lorenz96"] = {
       "sigma_y":[1, 0.7], 
       # "methods":["LETKF", "EnKF_Sqrt", "EnKF_PertObs", "iEnKS"],
       "methods":["EnKF", "ESRF", "LETKF", "iEnKS-PertObs"],
       "N_list":[5,10,15,20,40,60,100],
       "infl_list":L96_INFL,
       "letkf_infl_list":L96_INFL_LETKF,
       "loc_rad_list":L96_LOC,
       "num_trials": 1
       }

# ####################### L63
L63_INFL = {5: [1 + 0.02*i for i in range(11)],
       10: [1 + 0.02*i for i in range(11)],
       15: [1 + 0.02*i for i in range(11)],
       20: [1 + 0.01*i for i in range(11)],
       40: [1 + 0.01*i for i in range(11)],
       60: [1 + 0.01*i for i in range(6)],
       100: [1 + 0.01*i for i in range(6)],
       }

GRID_SEARCH_INFO["lorenz63"] = {
       "sigma_y":[1, 0.7], 
       # "methods":["EnKF_Sqrt", "EnKF_PertObs", "iEnKS"],
       "methods":["EnKF", "ESRF", "iEnKS-PertObs"],
       "N_list":[5,10,15,20,40,60,100],
       "infl_list":L63_INFL,
       "loc_rad_list":[],
       "num_trials": 1
       }

#sort grid search info by reverse alphabetical order of keys, so we have L96, L63, ks
GRID_SEARCH_INFO = dict(sorted(GRID_SEARCH_INFO.items(), key=lambda item: item[0], reverse=True))