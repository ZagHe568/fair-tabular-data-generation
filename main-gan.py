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
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
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

    df_train_data_neutral = df_train_data.drop(columns=['sex', 'relationship'])
    df_train_data_sens = df_train_data[['sex', 'relationship']]

    from ctgan import CTGANSynthesizer
    ctgan = CTGANSynthesizer()

    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
    #     'sex',
        'native-country',
        'Y'
    ]

    print('Training neutral features GAN....')
    ctgan.fit(df_train_data_neutral, discrete_columns, epochs=1000)
    print('Done')
    df_generated_neutral = ctgan.sample(df_train_data.shape[0])

    print('Training ')
    ctgan2 = CTGANSynthesizer()
    discrete_columns2 = [
        #     'workclass',
        #     'education',
        #     'marital-status',
        #     'occupation',
        'relationship',
        #     'race',
        'sex',
        #     'native-country',
        #     'Y'
    ]
    ctgan2.fit(df_train_data_sens, list(df_train_data_sens.columns), epochs=1000)
    df_generated_gendered = ctgan2.sample(df_train_data.shape[0])

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelBinarizer


    def preprocessing(a, columns):
        a_processed = pd.get_dummies(a, columns=columns)
        import sklearn
        scaler = sklearn.preprocessing.StandardScaler()
        a_scaled = scaler.fit_transform(a_processed)
        return a_scaled


    df_train_gen_neutral_processed = preprocessing(pd.concat([df_train_data_neutral, df_generated_neutral]),
                                                   ['workclass', 'education', 'marital-status',
                                                    'occupation', 'race', 'native-country', 'Y'])
    df_train_neutral_processed = df_train_gen_neutral_processed[:df_train_data_neutral.shape[0]]
    df_gen_neutral_processed = df_train_gen_neutral_processed[df_train_data_neutral.shape[0]:]

    df_train_data_gender_processed = pd.get_dummies(df_train_data['sex'], columns=['sex']).drop(columns=['Male'])

    clf = LogisticRegression(random_state=0).fit(df_train_neutral_processed, df_train_data_gender_processed.values)
    gender_pred = clf.predict(df_gen_neutral_processed)

    def negative_pick(pred):
        mask = df_generated_gendered['sex'] == ('Male' if pred == 1 else 'Female')
        return df_generated_gendered[mask].iloc[np.random.randint(mask.sum())]


    gendered_pred = pd.DataFrame(gender_pred).iloc[:, 0].apply(negative_pick)
    df_generated = pd.concat([df_generated_neutral, gendered_pred], axis=1)
    df_generated = df_generated[list(df_train_data.columns)]

    top_k_points = df_generated[:int(args.k * df_generated.shape[0])]
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
    df_train_data_male = df_train_data[df_train_data['sex'] =='Male']
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
    # print(mask_wrong_cases.sum())