HIGHLY_CORRELATED_FEATURES = ['Perimeter (um)', # highly correlated with 'Area (um2)'
            'Equivalent diameter (um)', # highly correlated with 'Area (um2)'
            'Equivalent volume (um3)', # highly correlated with 'Area (um2)'
            'Normalized sum intensity', # highly correlated with 'Area (um2)'
            'Normalized median intensity', # highly correlated with 'Normalized mean intensity'
            'Normalized upper quartile intensity', # highly correlated with 'Normalized mean intensity'
            'Normalized lower quartile intensity', # highly correlated with 'Normalized mean intensity'
            'Solidity', # highly correlated with 'Form factor'
            'Eccentricity', # highly correlated with 'Form factor'
            'Angular second moment', # highly correlated with 'Energy'
            'Dissimilarity', # highly correlated with 'Contrast'
            'Homogeneity', # highly correlated with 'Contrast'
            'Granularity', # highly correlated with 'Area (um2)'
]

FEATURE_SELECTION_CONDITIONS_MAPPING = {
    'CytoMix10': 'CytoMix',
    'CytoMix20': 'CytoMix',
    'Control': 'Control',
}

FEATURE_SELECTION_DEFAULT_CYTOKINE_DAYS = ['T2', 'T3', 'T4']

FEATURE_SELECTION_FEATURES_OF_INTEREST = ['Area (um2)', 'Normalized mean intensity']

IORS_TEST_CONDITIONS = [
        'CytoMix10',
        'CytoMix10 + DEX PREV',
        'CytoMix10 + DEX TREAT',
        'CytoMix10 + NAM PREV 0.5',
        'CytoMix10 + NAM PREV 1.0',
        'CytoMix10 + NAM TREAT 0.5',
        'CytoMix10 + NAM TREAT 1.0',
        'CytoMix10 + BHB PREV 0.5',
        'CytoMix10 + BHB PREV 1.0',
        'CytoMix10 + BHB TREAT 0.5',
        'CytoMix10 + BHB TREAT 1.0',
]

IORS_DAYS_OF_INTEREST = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']

IORS_FEATURES_OF_INTEREST = ['Area (um2)', 'Normalized mean intensity']

IORS_HELPER_COLUMNS = ['Well', 'Organoid_index', 'Day', 'Condition']

MATPLOTLIB_PRISM_STYLE = {
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 1,
        'legend.edgecolor': 'black',
        'legend.fancybox': False
}