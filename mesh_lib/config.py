from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATA_IN = edict()
__C.DATA_IN.DATA_FOLDER = r"/Users/jaeikjeon/Workspace/CODE/2023/echo4d/template_ssm_resultsheart_seg"
__C.DATA_IN.SAVE_FOLDER = r"/Users/jaeikjeon/Workspace/CODE/2023/echo4d/template_ssm_resultsheart_seg_save"

#
# Dataset Config
#
# The following are the labels used in the generation of the meshes from https://zenodo.org/record/4506930#.Yt_sZtLMJ1M
# 01. LV myocardium (endo + epi)
# 02. RV myocardium (endo + epi)
# 03. LA myocardium (endo + epi)
# 04. RA myocardium (endo + epi)
# 05. Aorta
# 06. Pulmonary artery
# 07. Mitral valve
# 08. Triscupid valve
# 09. Aortic valve
# 10. Pulmonary valve
# 11. Appendage
# 12. Left superior pulmonary vein
# 13. Left inferior pulmonary vein
# 14. Right inferior pulmonary vein
# 15. Right superior pulmonary vein
# 16. Superior vena cava
# 17. Inferior vena cava
# 18. Appendage border
# 19. Right inferior pulmonary vein border
# 20. Left inferior pulmonary vein border
# 21. Left superior pulmonary vein border
# 22. Right superior pulmonary vein border
# 23. Superior vena cava border
# 24. Inferior vena cava border
# Define names for each of the tags in the model.


__C.LABELS = edict()
__C.LABELS.LABEL_NAME = "elemTag"
__C.LABELS.LV = 1
__C.LABELS.RV = 2
__C.LABELS.LA = 3
__C.LABELS.RA = 4
__C.LABELS.AORTA = 5
__C.LABELS.PULMONARY_ARTERY = 6
__C.LABELS.MITRAL_VALVE = 7
__C.LABELS.TRICUSPID_VALVE = 8
__C.LABELS.AORTIC_VALVE = 9
__C.LABELS.PULMONARY_VALVE = 10
__C.LABELS.APPENDAGE = 11
__C.LABELS.LSPV = 12
__C.LABELS.LIPV = 13
__C.LABELS.RSPV = 14
__C.LABELS.RIPV = 15
__C.LABELS.SUPERIOR_VENA_CAVA = 16
__C.LABELS.INFERIOR_VENA_CAVA = 17
__C.LABELS.APPENDAGE_BORDER = 18
__C.LABELS.RIPV_BORDER = 19
__C.LABELS.LIPV_BORDER = 20
__C.LABELS.LSPV_BORDER = 21
__C.LABELS.RSPV_BORDER = 22
__C.LABELS.SUPERIOR_VENA_CAVA_BORDER = 23
__C.LABELS.INFERIOR_VENA_CAVA_BORDER = 24

__C.DATA_OUT = edict()
__C.DATA_OUT.SELECTED_VIEWS = [
    # "rv_inflow",
    "PLAX",  # lv_plax",
    "PSAX AV level",  # "psax_aortic",
    "PSAX basal",  # "psax_mv",
    "PSAX mid",  # "psax_pm",
    "PSAX apex",  # "psax_lower",
    "A4CH",  # "a4c",
    # "a5c",
    "A2CH",  # "a2c",
    "A3CH",  # "a3c",
]
__C.DATA_OUT.SAVE_IMG_RESOLUTION = (224, 224)  # Hasn't been verified for aspect ratios other than 1:1
__C.DATA_OUT.SAVE_BCKGD_CLR = "black"
__C.DATA_OUT.FANCY_PLOT = False

__C.PARAMETERS = edict()
__C.PARAMETERS.SUBSAMPLE_FACTOR = 20  # How much to subsample mesh for the low resolution, fast calculations
__C.PARAMETERS.THRESHOLD_PERCENTAGE = 0.7  # Percentage of mesh to remove in order to speed up finding LV apex
__C.PARAMETERS.NUM_WORKERS = 8  # Number of CPU cores to use for multiprocessing
