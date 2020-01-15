import warnings
warnings.filterwarnings("ignore")

import sys, os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, precision_score



base_dir = r'C:\Users\kotov-d\Documents\BASES'
features_path = os.path.join(base_dir, 'telecom_vad', 'feature', 'opensmile')


with open(os.path.join(features_path, 'x_train.pkl'), 'rb') as f:
    x_train = np.vstack(pickle.load(f))
with open(os.path.join(features_path, 'x_test.pkl'), 'rb') as f:
    x_test = np.vstack(pickle.load(f))
with open(os.path.join(features_path, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f).loc[:, 'cur_label']
with open(os.path.join(features_path, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f).loc[:, 'cur_label']


# use lightgbm to get top-100 features
#================================================================================================
#================================================================================================

clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
                         objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)

clf.fit(x_train, y_train)

print(clf.feature_importances_)

dict_importance = {}
for feature, importance in zip(range(len(clf.feature_importances_)), clf.feature_importances_):
    dict_importance[feature] = importance

best_features = []

for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
    if idx == 10:
        break
    best_features.append(w)

print(best_features)
#
# with open(os.path.join(r'C:\Users\kotov-d\Documents', 'clf' + '.pkl'), 'wb') as f:
#     pickle.dump(clf, f, protocol=2)
#
# # with open(os.path.join(r'C:\Users\kotov-d\Documents', 'clf' + '.pkl'), 'rb') as f:
# #     clf = pickle.load(f)

# # вывод предсказанных и реальных значений
# print(clf.classes_.tolist()+['pred','y'])
# print(pd.DataFrame(data=np.hstack((clf.predict_proba(x_test), clf.predict(x_test).reshape(-1,1),
#                                   y_test.values.reshape(-1,1))), columns=clf.classes_.tolist()+['pred','y']))


# use mutual correlation to get rid of features
#================================================================================================
#================================================================================================

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

vecs = []
for i in range(x_train.shape[1]):
    vecs.append(x_train[:,i])
corr_matrix = pd.DataFrame(np.corrcoef(vecs))

# почему-то 0 и 1512 фича считаются некорректно
# def func(x):
#     return x

corr_matrix = abs(corr_matrix.drop(columns=[0,1512], index=[0,1512]))


# # смотрим сколько значений корреляции выше трешхолда
# threshold = 0.9
# print((corr_matrix[corr_matrix>threshold].count().sum()-corr_matrix.shape[0])//2, "корреляций")
# print(round((corr_matrix[corr_matrix>threshold].count().sum()-2269)/(2269**2)*100, 3), "%")



# # отрисовка дендограммы
# Z = linkage(corr_matrix, 'single')
# plt.figure(figsize=(25, 25))
# dn = dendrogram(Z)
# plt.show()



# разбиваем фичи по кластерам
feature_indexes = list(range(x_train.shape[1]))
del feature_indexes[0]
del feature_indexes[1512]

clustering = AgglomerativeClustering(n_clusters=10).fit(corr_matrix)

clustering_labels = pd.DataFrame(columns=['feature', 'label'])
clustering_labels['feature'] = feature_indexes
clustering_labels['label'] = clustering.labels_
with open(r"C:\Users\kotov-d\Documents\TASKS\top_features\clustering_labels.pkl", "wb") as f:
    pickle.dump(clustering_labels, f)




# находим индексы лучших фичей
with open(r"C:\Users\kotov-d\Documents\TASKS\top_features\clustering_labels.pkl", "rb") as f:
    clustering_labels = pickle.load(f)

best_indexes = []

grouped = clustering_labels.groupby('label')
for name,group in grouped:
   curr_features = group.feature.to_list()
   group_corr_matrix = corr_matrix[corr_matrix.index.isin(curr_features)].loc[:,curr_features]
   group_corr_matrix = group_corr_matrix.assign(summ=lambda x: group_corr_matrix.sum())
   max_index = group_corr_matrix.summ.idxmax()
   best_indexes.append(max_index)

print(best_indexes)



# проверка полученных результатов
best_matrix = corr_matrix[corr_matrix.index.isin(best_indexes)]
# print(best_matrix)
print((best_matrix.sum().sum()-best_matrix.shape[0])/(best_matrix.shape[0]**2-best_matrix.shape[0]))
print((corr_matrix.sum().sum()-corr_matrix.shape[0])/(corr_matrix.shape[0]**2-corr_matrix.shape[0]))



# проверка с помощью обучения модели
pred = clf.fit(x_train[:,best_indexes], y_train).predict(x_test[:,best_indexes])
print("на фичах полученных при помощи корреляции f1 macro {}".format(round(f1_score(y_test, pred, average='macro'),3)))

pred = clf.fit(x_train[:,best_features], y_train).predict(x_test[:,best_features])
print("на фичах полученных при помощи feature_importances f1 macro {}".format(round(f1_score(y_test, pred, average='macro'),3)))



