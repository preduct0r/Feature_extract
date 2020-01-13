import warnings
warnings.filterwarnings("ignore")

import sys, os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, precision_score



base_dir = r'C:\Users\kotov-d\Documents\BASES'
features_path = os.path.join(base_dir, 'telecom_vad', 'feature', 'opensmile')


with open(os.path.join(features_path, 'x_train.pkl'), 'rb') as f:
    x_train = pickle.load(f)
with open(os.path.join(features_path, 'x_test.pkl'), 'rb') as f:
    x_test = pickle.load(f)
with open(os.path.join(features_path, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f).loc[:, 'cur_label']
with open(os.path.join(features_path, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f).loc[:, 'cur_label']



#================================================================================================
#================================================================================================

clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
                         objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)

clf.fit(x_train, y_train)

print(clf.feature_importances_)

# dict_importance = {}
# for feature, importance in zip(range(len(clf.feature_importances_)), clf.feature_importances_):
#     dict_importance[feature] = importance
#
# best_features = []
#
# for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
#     if idx == 100:
#         break
#     best_features.append(w)
#
# with open(os.path.join(r'C:\Users\kotov-d\Documents\check_relabling', 'clf' + '.pkl'), 'wb') as f:
#     pickle.dump(clf, f, protocol=2)
# #
# with open(os.path.join(r'C:\Users\kotov-d\Documents\check_relabling', 'data_iemo' + '.pkl'), 'wb') as f:
#     pickle.dump([iemocap_x_train, iemocap_y_train, iemocap_x_test, iemocap_y_test], f, protocol=2)
#
# with open(os.path.join(r'C:\Users\kotov-d\Documents\check_relabling', 'clf' + '.pkl'), 'rb') as f:
#     clf = pickle.load(f)
#
# with open(os.path.join(r'C:\Users\kotov-d\Documents\check_relabling', 'data_iemo' + '.pkl'), 'rb') as f:
#     [iemocap_x_train, iemocap_y_train, iemocap_x_test, iemocap_y_test] = pickle.load(f)
#
# print(clf.classes_.tolist()+['pred','y'])
#
# print(pd.DataFrame(data=np.hstack((clf.predict_proba(iemocap_x_test), clf.predict(iemocap_x_test).reshape(-1,1),
#                                   iemocap_y_test.values.reshape(-1,1))), columns=clf.classes_.tolist()+['pred','y']))