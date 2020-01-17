import warnings
warnings.filterwarnings("ignore")

import sys, os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, precision_score
from sklearn.metrics import pairwise_distances




base_dir = r'C:\Users\kotov-d\Documents\BASES'
features_path = os.path.join(base_dir, 'telecom_vad', 'feature', 'opensmile')


with open(os.path.join(features_path, 'x_train.pkl'), 'rb') as f:
    x_train = pd.DataFrame(np.vstack(pickle.load(f)))
with open(os.path.join(features_path, 'x_test.pkl'), 'rb') as f:
    x_test = pd.DataFrame(np.vstack(pickle.load(f)))
with open(os.path.join(features_path, 'y_train.pkl'), 'rb') as f:
    y_train = pickle.load(f).loc[:, 'cur_label']
with open(os.path.join(features_path, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f).loc[:, 'cur_label']



x_train['target'] = y_train
x_test['target'] = y_test

x_train = x_train[x_train.target!='defective'][x_train.target!='sad'][x_train.target!='not_ informative']
y_train = x_train.target
x_train.drop(columns=['target'], inplace=True)

x_test = x_test[x_test.target!='defective'][x_test.target!='sad'][x_test.target!='not_ informative']
y_test = x_test.target
x_test.drop(columns=['target'], inplace=True)


#================================================================================================
#================================================================================================
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


vecs = []
for i in range(x_train.shape[1]):
    vecs.append(x_train.iloc[:,i])
corr_matrix = pd.DataFrame(np.corrcoef(vecs))

# почему-то 0 и 1512 фича считаются некорректно
def func(x):
    return 1-abs(x)

corr_matrix = func(corr_matrix.drop(columns=[0,1512], index=[0,1512]))


# разбиваем фичи по кластерам
feature_indexes = list(range(x_train.shape[1]))
del feature_indexes[0]
del feature_indexes[1511]



# Делаем кастомную affinity func
# def sim(x, y):
#     return corr_matrix.loc[int(x[0]),int(y[0])]
#
# def sim_affinity(X):
#     return pairwise_distances(X, metric=sim)
#

# # Обучаем Agglomerative Clustering с новым affinity
# cluster = AgglomerativeClustering(n_clusters=10, affinity=sim_affinity, linkage='average')
# cluster.fit(np.array(feature_indexes).reshape(-1, 1))

# # Обучаем Agglomerative Clustering с "precomputed"
# cluster = AgglomerativeClustering(n_clusters=10, affinity='precomputed', linkage='average')
# cluster.fit(corr_matrix)

# Обучаем DBSCAN с новым affinity
cluster = DBSCAN(eps=0.03, min_samples=2, metric="precomputed")
cluster.fit(corr_matrix)

# Делаю сводную таблицу соответствия фичей и лэйблов
clustering_labels = pd.DataFrame(columns=['feature', 'label'])
clustering_labels['feature'] = feature_indexes
clustering_labels['label'] = cluster.labels_


temp = pd.DataFrame(columns=['label', 'value_count'])
temp['value_count'] = clustering_labels.label.value_counts()
temp['label'] = clustering_labels.label.value_counts().index



# Находим самую репрезентативную фичу в каждом кластере, делаем сводную таблицу по кластерам
# и матрицу корреляции для отобранных фичей
best_indexes = []
grouped = clustering_labels.groupby('label')
df_clusters_info = pd.DataFrame(columns=['mean','max','std','label'])
df_clusters_info['label'] = clustering_labels.label.unique()
not_corr_idxs = []
for name,group in grouped:
   curr_features = group.feature.to_list()
   group_corr_matrix = corr_matrix[corr_matrix.index.isin(curr_features)].loc[:,curr_features]
   summ = group_corr_matrix.sum()
   maxx = group_corr_matrix.max().max()
   meann = group_corr_matrix.mean().mean()                                # для демонстрации каждого кластера
   stdd = group_corr_matrix.std().mean()
   curr_idx = df_clusters_info[df_clusters_info.label == name].index[0]
   df_clusters_info.loc[curr_idx,:] = [round(meann,3), round(maxx,3), round(stdd,3), name]
   max_index = summ.idxmax()
   if name==-1:
       not_corr_idxs = list(group_corr_matrix.index)
   else:
       best_indexes.append(max_index)
best_indexes += not_corr_idxs
best_vecs = [vecs[x] for x in best_indexes]
zz = pd.DataFrame(data=abs(np.corrcoef(best_vecs)), index=best_indexes, columns=best_indexes)
zz.to_csv(r"C:\Users\kotov-d\Documents\TASKS\task#7\best_features_corr_matrix.csv", index=True)
df_clusters_info = df_clusters_info.merge(temp, how='left', on='label', left_index=True)
df_clusters_info = df_clusters_info.loc[:,['label','value_count','max','mean','std']]
df_clusters_info.to_csv(r"C:\Users\kotov-d\Documents\TASKS\task#7\df_clusters_info.csv", index=False)
# print(pd.read_csv(r"C:\Users\kotov-d\Documents\TASKS\task#7\best_features_corr_matrix.csv", index_col=0).iloc[:10,:10])







# use lightgbm to get top features
#================================================================================================
#================================================================================================
cur_pred = 0
def test_100_features(x_train, y_train, x_test, y_test):
    clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
                             objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)
    clf.fit(x_train, y_train)

    dict_importance = {}
    for feature, importance in zip(range(len(clf.feature_importances_)), clf.feature_importances_):
        dict_importance[feature] = importance

    best_lgbm_features = []

    for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
        if idx == 1:
            break
        best_lgbm_features.append(w)

    clf.fit(x_train.iloc[:,best_lgbm_features], y_train)

    pred = clf.predict(x_test.iloc[:,best_lgbm_features])


    print(round(f1_score(y_test, pred, average='macro'),30))

test_100_features(x_train, y_train, x_test, y_test)
test_100_features(x_train.iloc[:,best_indexes], y_train, x_test.iloc[:,best_indexes], y_test)
print(y_train.value_counts())




# # вывод предсказанных и реальных значений
# print(clf.classes_.tolist()+['pred','y'])
# print(pd.DataFrame(data=np.hstack((clf.predict_proba(x_test), clf.predict(x_test).reshape(-1,1),
#                                   y_test.values.reshape(-1,1))), columns=clf.classes_.tolist()+['pred','y']))



# ====================================================================================================================
# ====================================================================================================================
# # проверка полученных результатов с помощью усреднения значений ячеек таблицы корреляции
# best_matrix = corr_matrix[corr_matrix.index.isin(best_indexes)]
# print((best_matrix.sum().sum()-best_matrix.shape[0])/(best_matrix.shape[0]**2-best_matrix.shape[0]))
# print((corr_matrix.sum().sum()-corr_matrix.shape[0])/(corr_matrix.shape[0]**2-corr_matrix.shape[0]))



# проверка с помощью обучения модели на разных фичах
from sklearn.svm import SVC
clf=SVC()
pred = clf.fit(x_train.iloc[:,best_indexes], y_train).predict(x_test.iloc[:,best_indexes])
print("на фичах полученных при помощи корреляции f1 macro {}".format(round(f1_score(y_test, pred, average='macro'),3)))

# pred = clf.fit(x_train[:,best_features], y_train).predict(x_test[:,best_features])
# print("на фичах полученных при помощи feature_importances f1 macro {}".format(round(f1_score(y_test, pred, average='macro'),3)))

pred = clf.fit(x_train, y_train).predict(x_test)
print("на оригинальных фичах f1 macro {}".format(round(f1_score(y_test, pred, average='macro'),3)))



# возможно прегодится
# ====================================================================================================================
# ====================================================================================================================

# # смотрим сколько значений корреляции выше трешхолда
# threshold = 0.9
# print((corr_matrix[corr_matrix>threshold].count().sum()-corr_matrix.shape[0])//2, "корреляций")
# print(round((corr_matrix[corr_matrix>threshold].count().sum()-2269)/(2269**2)*100, 3), "%")



# # отрисовка дендограммы
# Z = linkage(corr_matrix, 'single')
# plt.figure(figsize=(25, 25))
# dn = dendrogram(Z)
# plt.show()
