import torch

class Config:
    # System Parameters (Eq 11)
    SIGMA_PROCESS = 0.25
    MU = 10.0
    RHO = 28.0
    TAU = 8.0 / 3.0
    
    # Observation Parameters (Eq 12)
    SIGMA_OBS = 0.25 # gamma
    
    # Simulation Parameters
    DT_SIM = 0.01   
    
    # Data Parameters
    N_TRAIN_TRAJ = 1024
    LEN_TRAIN_TRAJ = 1024
    N_TEST_TRAJ = 64
    LEN_TEST_TRAJ = 15
    L_LAG = 1 
    
    # Training Hyperparameters
    BATCH_SIZE = 256 
    LR = 0.005
    EPOCHS = 23000
    HIDDEN_DIM = 256
    LAYERS = 5
    EMBED_DIM = 4 
    
    # Inference Hyperparameters
    MC_SAMPLES_J = 21
    GUIDANCE_STEP_ZETA = 0.0002
    
    # Interpolant Parameters
    N_INTERPOLANT_STEPS = 100 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"