from lightgbm import LGBMClassifier
lgbm_c = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0, 
            learning_rate=0.5, max_depth=7, min_child_samples=20, 
            min_child_weight=0.001, min_split_gain=0.0, n_estimators=100, 
            n_jobs=-1, num_leaves=500, objective='binary', random_state=None, 
            reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0, 
            subsample_for_bin=200000, subsample_freq=0)