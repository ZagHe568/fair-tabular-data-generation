import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k', type=float, default=0.4)
    parser.add_argument('--gpu', type=str, default='')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # df = pd.read_csv('adult', sep=', ', engine='python')
    # data = df.values
    #
    # train_data, test_data = train_test_split(data, test_size=0.3, shuffle=True, random_state=seed)

    # df_train_data = pd.DataFrame(train_data, columns=df.columns)
    # df_test_data = pd.DataFrame(test_data, columns=df.columns)

    df_train_data = pd.read_csv('adult_known.data.txt', sep=', ', engine='python')
    df_test_data = pd.read_csv('adult_known.test.txt', sep=', ', engine='python')

    df_generated = df_train_data.copy()
    df_generated['sex'] = df_generated['sex'].map({'Male': 'Female', 'Female': 'Male'})


    def preprocessing(a):
        a_processed = pd.get_dummies(a, columns=['workclass', 'education', 'marital-status',
                                                 'occupation', 'relationship', 'race', 'sex', 'native-country', 'Y'])
        import sklearn
        scaler = sklearn.preprocessing.StandardScaler()
        a_scaled = scaler.fit_transform(a_processed)
        return a_scaled


    def get_simlarity_utils(df_generated_group, df_train_group):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=10, random_state=seed).fit(df_train_group)
        centroids = kmeans.cluster_centers_
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(df_generated_group, centroids)  # (df_generated_group.shape[0],10)
        max_distances = 1. / distances.max(axis=1)
        return max_distances


    def get_similarity(df_generated_data, df_train):
        sims = np.zeros(df_generated.shape[0])
        for sex in ['Male', 'Female']:
            for y in ['<=50K', '>50K']:
                mask_generated_group = (df_generated_data['sex'] == sex) & (df_generated_data['Y'] == y)
                df_generated_group = df_generated_data.loc[mask_generated_group]
                df_train_group = df_train[(df_train['sex'] == sex) & (df_train['Y'] == y)]
                df_preprocessed = preprocessing(pd.concat([df_generated_group, df_train_group]))
                df_generated_preprocessed = df_preprocessed[:df_generated_group.shape[0]]
                df_train_preprocessed = df_preprocessed[df_generated_group.shape[0]:]
                max_distances = get_simlarity_utils(df_generated_preprocessed, df_train_preprocessed)
                sims[mask_generated_group] = max_distances
        return sims


    def top_k_data_points(df_generated, df_train, k):
        sims = get_similarity(df_generated, df_train)
        idxes = np.argsort(-sims)[:int(df_generated.shape[0] * k)]

        return df_generated.iloc[idxes]

    top_k_points = top_k_data_points(df_generated, df_train_data, args.k)
    df_train_data_new = pd.concat([df_train_data, top_k_points])

    df_all = pd.concat([df_train_data_new, df_test_data])
    df_x = df_all.drop(columns=['Y'])
    df_y = df_all['Y']
    df_processed = pd.get_dummies(df_x, columns=['workclass', 'education', 'marital-status',
                                                 'occupation', 'relationship', 'race', 'sex', 'native-country'])
    df_processed_y = pd.get_dummies(df_y, columns=['Y']).drop(columns=['>50K'])
    import sklearn

    scaler = sklearn.preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(df_processed)
    df_scaled_x_train = df_scaled[:df_train_data_new.shape[0]]
    df_scaled_x_test = df_scaled[df_train_data_new.shape[0]:]
    df_y_train = df_processed_y[:df_train_data_new.shape[0]]
    df_y_test = df_processed_y[df_train_data_new.shape[0]:]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
    clf = LogisticRegression(random_state=0).fit(df_scaled_x_train, df_y_train)
    y_pred = clf.predict(df_scaled_x_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(df_y_test, y_pred)
    print(f'acc: {acc:.3f}')


    def stat_parity(y_pred, genders):
        import numpy as np
        gender1_idx = genders == 'Male'
        gender2_idx = genders == 'Female'
        pos_idx_gender1 = np.where(y_pred[gender1_idx] == 1)[0].astype(np.int32)
        pos_idx_gender2 = np.where(y_pred[gender2_idx] == 1)[0].astype(np.int32)
        prob1 = pos_idx_gender1.shape[0] / gender1_idx.sum()
        prob2 = pos_idx_gender2.shape[0] / gender2_idx.sum()
        return prob1 - prob2


    def eq_oppo_bianry(preds, sens, labels):
        g1_idx = sens == 'Male'
        g2_idx = sens == 'Female'

        pos_label_idx = labels == 1
        pos_pred_idx = preds == 1

        denominator1 = (g1_idx & pos_label_idx).sum()
        numerator1 = (g1_idx & pos_label_idx & pos_pred_idx).sum()

        denominator2 = (g2_idx & pos_label_idx).sum()
        numerator2 = (g2_idx & pos_label_idx & pos_pred_idx).sum()

        oppo1 = numerator1 / denominator1 if denominator1 != 0 else 0
        oppo2 = numerator2 / denominator2 if denominator2 != 0 else 0

        return oppo1 - oppo2


    stat_p = np.abs(stat_parity(y_pred, df_test_data['sex'].values))
    eq_op = np.abs(eq_oppo_bianry(y_pred, df_test_data['sex'].values, df_y_test.values[:, 0]))
    print(f'stat_p: {stat_p:.3f}, eq_op: {eq_op: .3f}')

    from scipy import stats
    def calc_entropy(data):
        entropies = []
        for column in data.columns:
            p_data = data[column].value_counts()
            entropies.append(round(stats.entropy(p_data), 2))
        entropies.insert(0, round(sum(entropies), 2))
        return entropies[0]
    df_train_data_male = df_train_data[df_train_data['sex']=='Male']
    df_train_data_female = df_train_data[df_train_data['sex'] == 'Female']
    df_train_data_new_male = df_train_data_new[df_train_data_new['sex']=='Male']
    df_train_data_new_female = df_train_data_new[df_train_data_new['sex'] == 'Female']
    entropy_male = calc_entropy(df_train_data_male)
    entropy_female = calc_entropy(df_train_data_female)
    entropy_new_male = calc_entropy(df_train_data_new_male)
    entropy_new_female = calc_entropy(df_train_data_new_female)
    print(f'entropy_male:{entropy_male}, entropy_female:{entropy_female}, entropy_gap:{entropy_male-entropy_female}')
    print(f'entropy_new_male:{entropy_new_male}, entropy_female:{entropy_new_female}, entropy_new_gap:{entropy_new_male - entropy_new_female}')

    # mask_wrong_cases = ((df_train_data_new['relationship'] == 'Wife') & (df_train_data_new['sex'] == 'Male')) \
    #                    | ((df_train_data_new['relationship'] == 'Husband') & (df_train_data_new['sex'] == 'Female'))
    # # print(df_train_data_new[mask_wrong_cases])
    # print(mask_wrong_cases.sum())