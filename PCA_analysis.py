from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
RANDOM_STATE = 2137


class PCA_analyzer():
    def __init__(self, df, max_n_of_features):
        self.max_n_of_features = max_n_of_features
        self.df = df
        self.df = StandardScaler().fit_transform(self.df)

    def get_variance_ratio(self, i):
        pca = PCA(n_components=i)
        pca.fit_transform(self.df)
        return pca.explained_variance_ratio_.cumsum()[-1:]

    def get_pca_variance_cumsum(self):
        components_table = np.zeros(self.max_n_of_features)
        for i in range(self.max_n_of_features):
            vr = self.get_variance_ratio(i)
            if (len(vr) > 0):
                components_table[i] = vr
        return components_table

    def get_pca_summary(self):

        pca = PCA(
            random_state=RANDOM_STATE, n_components=self.max_n_of_features
        )

        data_transformed = pca.fit_transform(self.df)

        n_components = pca.components_.shape[1]
        pca_cols = [f'PC{x}' for x in range(n_components)]

        df_transformed = pd.DataFrame(
            data=data_transformed,
            columns=pca_cols[:self.max_n_of_features]
        )
        # Principal components correlation coefficients
        df_pca_components = pd.DataFrame(
            data=np.transpose(pca.components_),
            columns=pca_cols[:self.max_n_of_features],
        )

        pca_dict = {
            'Proportion of Variance': pca.explained_variance_ratio_,
            'Cumulative Proportion': np.cumsum(pca.explained_variance_ratio_)
        }

        # summary
        df_pca_summarize = pd.DataFrame.from_dict(
            data=pca_dict,
            columns=pca_cols[:self.max_n_of_features],
            orient='index'
        )

        return df_transformed, df_pca_components, df_pca_summarize

    def ggbiplot_pca(self, score, coeff, labels=None):
        plt.figure(figsize=(10, 8))
        xs = score[:, 0]
        ys = score[:, 1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.scatter(xs * scalex, ys * scaley, marker='.', alpha=0.5)
        for i in range(n):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=1)
            if labels is None:
                plt.text(
                    coeff[i, 0] * 1.15,
                    coeff[i, 1] * 1.15,
                    "Var"+str(i+1),
                    color='g', ha='center', va='center')
            else:
                plt.text(
                    coeff[i, 0] * 1.15,
                    coeff[i, 1] * 1.15,
                    labels[i],
                    color='g', ha='center', va='center')
        # plt.xlim(-0.25, 1)
        # plt.ylim(-0.5, 1)
        plt.xlabel(f"PC{0}")
        plt.ylabel(f"PC{1}")
        plt.grid()

    def show_pca_plot(self, labels):
        df_transformed, df_pca_components, pca_summarize = self.get_pca_summary()

        self.ggbiplot_pca(
            score=df_transformed.values[:, 0:2],
            coeff=np.transpose(df_pca_components.values)[0:2, :].T,
            labels=labels
        )
        plt.show()

    def print_pca_data(self, labels):
        print("Cumulative sum of variance:")
        print(self.get_pca_variance_cumsum())
        df_transformed, df_pca_components, pca_summarize = self.get_pca_summary()
        print("Transformed data: ")
        print(df_transformed)
        print("PCA components:")
        print(df_pca_components)
        print("Summary")
        print(pca_summarize)
        self.show_pca_plot(labels)
