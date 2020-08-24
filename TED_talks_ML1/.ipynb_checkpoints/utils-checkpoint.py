import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as skm
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestClassifier

#pio.renderers.default = "browser"

def load_data(data_path, target_feature):
    '''
    Function to load dataframe as csv. Split target variable and the rest, reorder dataframe and verify if NAs are contained.
    '''

    # Read csv
    df = pd.read_csv(data_path, usecols=[i for i in range(1, 13)],
                         dtype={'general_opinion': str, 'speaker_pop': str, 'event': str, 'location': str, 'published_year': str,
                                'published_month': str, 'tags': str})
    # split target feature form dataframe
    target = df[target_feature]
    cols_no_target = df.columns != target_feature
    df = df.loc[:, cols_no_target]

    # reorder dataframe: keep all numeric columns at the begining
    columns_types = df.dtypes
    numeric_cols = []
    non_numeric_cols = []
    for column in columns_types.index.tolist():
        if columns_types[column] == int:
            numeric_cols.append(column)
        else:
            non_numeric_cols.append(column)
    # new defined order
    ordered_columns = numeric_cols + non_numeric_cols
    df = df.loc[:, ordered_columns]
    # add target variable at the end
    df = pd.concat([df, target], axis=1)

    # find NAs if there were
    if df.isnull().any().any():
        print("NAs detected in dataframe!")

    df.columns = [item.upper() for item in df.columns.tolist()]

    return df

def read_processing(data_file, threshold):
    df = load_data(data_file, target_feature='views')

    ''' ...................................................................................................................
    Generamos variable respuesta categórica (para abrir abanico de posibilidades para posteriores tratamientos)
    Diferenciamos entre video popular_video y not_popular_video
    Un video se considerará popular si el numero de reproducciones es mayor que el 3er quantile de los datos de train
    ...................................................................................................................'''

    '''
    # Se muestra donde queda el q75 en la variable respuesta
    fig, ax = plt.subplots(2, 1)
    # subplot 1: whole target values range
    ax[0].hist(df['VIEWS'], bins=200)
    ax[0].axvline(x=threshold, color='k', linestyle='-')
    ax[0].set_title("Target variable, whole range [{}, {}] VIEWS".format(np.min(df['VIEWS']), np.max(df['VIEWS'])))
    # subplot 2: reduced range (from median to max)
    q50 = np.quantile(df['VIEWS'], 0.5)
    q100 = np.max(df['VIEWS'])
    ax[1].hist(df['VIEWS'], bins=200)
    ax[1].axvline(x=threshold, color='k', linestyle='-')
    ax[1].set_xlim([np.quantile(df['VIEWS'], 0.5), np.max(df['VIEWS'])])
    ax[1].set_xscale('log')
    ax[1].set_title(
        "Target variable, reduced range (median to max) - [{}, {}] Million VIEWS".format(round(q50 / (10 ** 6), 1),
                                                                                         round(q100 / (10 ** 6), 1)))
    '''

    # Creamos variable respuesta categórica
    target_categorical = df['VIEWS'].copy()
    target_categorical[:] = 'not_popular_video'
    ids_over_th = df['VIEWS'] >= threshold
    target_categorical[ids_over_th] = 'popular_video'
    target_categorical = target_categorical.rename(target_categorical.name + '_CATEGORICAL')

    # añadimos la nueva serie al dataframe
    df = pd.concat([df, target_categorical], axis=1)

    return df

def color_by_category(df_converted, df_categorical, sup_title = '', nrows=2, ncols=2, figsize=(12, 4)):
    '''
    Funcion que pinta un dataframe al que se le ha aplicado reducción de la dimensionalidad. Usa los colores en función
    de las diferentes categorías que tienen las variables categóricas (en df_categorical)
    '''
    # por cada variable pintamos de un color cada categoría
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    i = 0
    dict_subplots = get_dict_subplots(nrows, ncols)
    for feature in df_categorical.columns:
        feature_categories = df_categorical[feature].unique().tolist()
        #print(feature, feature_categories)
        for category in feature_categories:
            ids = df_categorical[feature] == category
            xvalues = df_converted.loc[ids, df_converted.columns[0]]
            yvalues = df_converted.loc[ids, df_converted.columns[1]]
            row = dict_subplots[i][0]
            col = dict_subplots[i][1]

            if (nrows==1) or (ncols==1):
                if (nrows == 1) and (ncols == 1):
                    ax.scatter(xvalues, yvalues, label=category, alpha=0.5)
                    ax.legend()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(df_converted.columns[0])
                    ax.set_ylabel(df_converted.columns[1])
                    ax.set_title("Colored by " + feature)
                else:
                    ax[i].scatter(xvalues, yvalues, label=category, alpha=0.5)
                    ax[i].legend()
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                    ax[i].set_xlabel(df_converted.columns[0])
                    ax[i].set_ylabel(df_converted.columns[1])
                    ax[i].set_title("Colored by " + feature)
            else:
                ax[col, row].scatter(xvalues, yvalues, label=category, alpha=0.5)
                ax[col, row].legend()
                ax[col, row].set_xticks([])
                ax[col, row].set_yticks([])
                ax[col, row].set_xlabel(df_converted.columns[0])
                ax[col, row].set_ylabel(df_converted.columns[1])
                ax[col, row].set_title("Colored by " + feature)
        i += 1

    if sup_title != '':
        fig.suptitle(sup_title, fontsize=14, y=0.99)


def get_dict_subplots(nrows, ncols):
    '''
    Function to assing the subplot coordinates to subplot number
    :param nrows: number of figure subplots rows
    :param ncols: number of figure subplots columns
    '''
    i = 0
    dict_out = {}
    for row in range(nrows):
        for col in range(ncols):
            dict_out[i] = [row, col]
            i += 1

    return dict_out


def plot_by_date(df_input, nrows, ncols):
    # Plot views variation per months for each year

    # set views to million views
    df_input.loc[:, 'VIEWS'] = df_input.loc[:, 'VIEWS'].copy().map(lambda x: round(x/(10**6), 2))
    years = df_input['PUBLISHED_YEAR'].unique().tolist()
    # convert published_month to integer in order to sort df
    df_input.loc[:, 'PUBLISHED_MONTH'] = df_input.loc[:, 'PUBLISHED_MONTH'].copy().astype(int)

    # Dictionary to manage subplots
    dict_subplots = get_dict_subplots(nrows=nrows, ncols=ncols)
    fig1, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    fig2, bx = plt.subplots(nrows, ncols, sharex=True)
    #fig1.tight_layout()
    #fig2.tight_layout()
    fig1.suptitle('Views (maximum value) per year and month')
    fig2.suptitle('Views (median value) per year and month')


    # ticks labels
    ticks = {}
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i in range(1, 13):
        ticks[i] = months[i-1]

    i = 0
    for year in years:
        df_year = df_input.loc[df_input['PUBLISHED_YEAR'] == year, :].sort_values('PUBLISHED_MONTH')
        mean_month = (df_year.groupby('PUBLISHED_MONTH')['VIEWS']
                      .mean()
                      .rename('MEAN_VIEWS'))
        median_month = (df_year.groupby('PUBLISHED_MONTH')['VIEWS']
                        .median()
                        .rename('MEDIAN_VIEWS'))
        max_month = (df_year.groupby('PUBLISHED_MONTH')['VIEWS']
                        .max()
                        .rename('MAX_VIEWS'))
        df_agg = pd.concat([mean_month, median_month, max_month], axis=1)

        # plot
        row = dict_subplots[i][0]
        col = dict_subplots[i][1]
        #ax[row, col].bar(x= df_agg.index, height=df_agg['MAX_VIEWS'], label='MAX_VIEWS', color='orange', alpha = 0.7)
        #bx[row, col].bar(x= df_agg.index, height=df_agg['MEDIAN_VIEWS'], label='MEDIAN_VIEWS', alpha = 0.7)

        ax[row, col].plot(df_agg.index, df_agg['MAX_VIEWS'], label='MAX_VIEWS', color='orange', alpha=0.7)
        bx[row, col].plot(df_agg.index, df_agg['MEDIAN_VIEWS'], label='MEDIAN_VIEWS', alpha=0.7)


        ax[row, col].set_xticklabels(ticks)
        bx[row, col].set_xticklabels(ticks)
        ax[row, col].set_title(year)
        bx[row, col].set_title(year)
        ax[row, col].set_ylabel("Views\n(millions)")
        bx[row, col].set_ylabel("Views\n(millions)")
        i += 1

    # add legend
    ax[0,0].legend()
    bx[0,0].legend()
    # change xticks
    ax[row, col].set_xticks([i for i in range(1,13)])
    ax[row, col].set_xticklabels(months)
    bx[row, col].set_xticks([i for i in range(1, 13)])
    bx[row, col].set_xticklabels(months)


def get_and_plot_umap(data_df, categorical_df, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', nrows=2, ncols=2, sup_title=''):

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    embedding = reducer.fit_transform(data_df)
    df_embedding = pd.DataFrame(embedding, columns=['UMAP-D1', 'UMAP-D2'])

    # plot UMAP
    title = 'n_neighbors: {} - min_dist: {} - metric: {}'.format(n_neighbors, min_dist, metric)
    color_by_category(df_embedding, categorical_df, nrows=nrows, ncols=ncols, sup_title=sup_title + '\n' + title)

def encord_var_cat (dataframe, column):
    enc = LabelEncoder()
    label_encoder = enc.fit(dataframe.loc[:, column])
    print("Categorical classes:", label_encoder.classes_)
    integer_classes = label_encoder.transform(label_encoder.classes_)
    print("Integer classes:", integer_classes)
    enc_var = pd.DataFrame({column: label_encoder.classes_, str(column + '_d'): integer_classes})
    dataframe.loc[:, column] = label_encoder.fit_transform(dataframe.loc[:, column])
    return enc_var

def dummies_var (train, dataframe, column):
    dummy_encord = encord_var_cat(train, column)
    result = pd.merge(dataframe, dummy_encord, how='inner',
                      on=[column, column])
    result = result.drop(column, axis=1)
    result = result.rename(columns={str(column + '_d'): column})
    return result


def normalise_dataframe(df_numeric: pd.DataFrame, df_categorical: pd.DataFrame):
    '''
        Normalises numeric features and convert categorical features to dummies using one-hot encoding
    '''

    # Normalización de variables numericas entre [0, 1]
    scaler = MinMaxScaler()
    norm_values = scaler.fit_transform(df_numeric.values)
    df_norm = pd.DataFrame(norm_values, columns=df_numeric.columns)

    # Unir ambos dfs
    df_out = pd.concat([df_norm, df_categorical], axis=1)

    return df_out


def get_new_names(original_columns, encoding_categories):
    # columns names
    i = 0
    colnames = []
    for bin_variables in encoding_categories:
        original_name = original_columns[i]
        i += 1
        for variable in bin_variables:
            new_name = original_name + "-" + variable
            colnames.append(new_name)
    return colnames

def get_models(models, selected_models):
    out_models = {}
    for model in models.keys():
        if model in selected_models:
            out_models[model] = models[model]
    return out_models

def get_ROC_metrics_nb(models, y_test):

    models_metrics = {}
    for model in models.keys():
        clf = models[model]['clf']
        X_test_cat = models[model]['X_test_cat']
        X_test_cont = models[model]['X_test_cont']
        y_pred_probs = clf.predict_proba(X_test_cat, X_test_cont)[:, 1]  # probabilities for 1s

        # Values for ROC curve
        fpr, tpr, ths = skm.roc_curve(y_test, y_pred_probs)
        auc = skm.roc_auc_score(y_test, y_pred_probs)
        metrics = {'fpr': fpr, 'tpr': tpr, 'thresholds': ths, 'auc': auc}
        models_metrics[model] = metrics

    return models_metrics

def get_ROC_metrics(models, y_test):

    models_metrics = {}
    for model in models.keys():
        clf = models[model]['clf']
        X_test = models[model]['X_test']
        y_pred_probs = clf.predict_proba(X_test)[:, 1]  # probabilities for 1s

        # Values for ROC curve
        fpr, tpr, ths = skm.roc_curve(y_test, y_pred_probs)
        auc = skm.roc_auc_score(y_test, y_pred_probs)
        metrics = {'fpr': fpr, 'tpr': tpr, 'thresholds': ths, 'auc': auc}
        models_metrics[model] = metrics

    return models_metrics

def plot_ROC(models_to_plot: dict, show_metrics=False, title ='', figsize=(6, 4), legendsize=12):

    plt.figure(figsize=figsize)
    for model in models_to_plot.keys():
        fp_rate = models_to_plot[model]['fpr']
        tp_rate = models_to_plot[model]['tpr']
        auc = models_to_plot[model]['auc']
        thresholds_raw = models_to_plot[model]['thresholds']
        thresholds = [round(th, 2) for th in thresholds_raw]
        if show_metrics:
            recall = models_to_plot[model]['tpr']
            #precision = models_to_plot[model]['precision']
            tnr = models_to_plot[model]['tnr']
            plt.plot(fp_rate, tp_rate, label="{} (auc:{}, recall:{}, tnr:{})".format(model, round(auc, 2), round(recall, 2), round(tnr, 2)))
        else:
            plt.plot(fp_rate, tp_rate, label= "{} ({})".format(model, round(auc, 2)))
            # for i in range(len(fp_rate)):
            #     plt.text(fp_rate[i]*0.8, tp_rate[i]*1.02, s=thresholds[i])
            #
            # if cool:
            #     fig = px.line(x=fp_rate, y=tp_rate, hover_name=thresholds)
            #     fig.show()

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve' + title)
    plt.legend(loc='lower right', fontsize=legendsize)
    plt.show()



def interactive_plot_ROC(models_to_plot: dict, title = '', figsize = (6, 4), render='browser'):

    pio.renderers.default = render
    # Create traces
    fig = go.Figure()

    for model in models_to_plot.keys():
        model_name = model.split('clf_')[1]
        fp_rate_raw = models_to_plot[model]['fpr']
        fp_rate = [round(i, 2) for i in fp_rate_raw]
        tp_rate_raw = models_to_plot[model]['tpr']
        tp_rate = [round(i, 2) for i in tp_rate_raw]
        auc = models_to_plot[model]['auc']
        thresholds_raw = models_to_plot[model]['thresholds']
        thresholds_text = [('<b>Threshold: ' + str(round(th, 2)) + '</b>') for th in thresholds_raw]
        #plt.plot(fp_rate, tp_rate, label=)
        fig.add_trace(go.Scatter(x=fp_rate,
                                 y=tp_rate,
                                 mode='lines',
                                 text= thresholds_text,
                                 name="<b>{}</b> ({})".format(model_name, round(auc, 2))))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='random model', line=dict(color='gray', dash='dash')))

    # Edit the layout
    fig.update_layout(title='ROC Curve' + title,
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate (Recall)')
    fig.show()




def pca_line_plot(pca_components, original_feature_names, title='', figsize=(16, 4)):

    x = [i for i in range(1, len(original_feature_names) +1)]    # values for x axis
    plt.figure(figsize=figsize)
    i = 1
    for component in pca_components:
        ks = abs(component) # absolute value of factors per variable
        plt.plot(x, ks, label="PC"+str(i), linewidth=4/(i+1), alpha=0.85)
        i += 1

    ticks_names = []
    for feature in original_feature_names:
        name = feature.replace('-', '\n')
        ticks_names.append(name)

    plt.grid()
    plt.legend()
    plt.xticks(x, ticks_names, rotation=30)
    plt.title("PCA" + title)
    plt.subplots_adjust(bottom=0.3)




def scale_dummies(train, test, x_exp=2):
    '''
    Replace 1s (for one-hot encoded variables) by:
       1/3  (where 3 is the number of different categories per feature)
       1/4  (where 4 is the number of different categories per feature)
    '''

    # Set categories to convert and number of categories per feature
    categoriesx3 = ['GENERAL_OPINION-negative', 'GENERAL_OPINION-neutral', 'GENERAL_OPINION-positive',
                    'SPEAKER_POP-not_popular', 'SPEAKER_POP-popular', 'SPEAKER_POP-very_popular']

    categoriesx4 = ['EVENT-Other', 'EVENT-TED', 'EVENT-TEDGlobal', 'EVENT-TEDx']

    map = {'categoriesx3': {1: 1 / (3**x_exp)},
           'categoriesx4': {1: 1 / (4**x_exp)}}

    # convert dummy variables from (0, 1) to (0, J)
    X_train_scaled = train.copy()
    X_test_scaled = test.copy()

    X_train_scaled.loc[:, categoriesx3] = X_train_scaled.loc[:, categoriesx3].replace(map['categoriesx3'])
    X_train_scaled.loc[:, categoriesx4] = X_train_scaled.loc[:, categoriesx4].replace(map['categoriesx4'])

    X_test_scaled.loc[:, categoriesx3] = X_test_scaled.loc[:, categoriesx3].replace(map['categoriesx3'])
    X_test_scaled.loc[:, categoriesx4] = X_test_scaled.loc[:, categoriesx4].replace(map['categoriesx4'])

    return X_train_scaled, X_test_scaled

def print_results(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print("------------------------------------------")
    print(classification_report(y_test, y_pred))
    print("------------------------------------------")
    print("accuracy", accuracy_score(y_test, y_pred))


def rf_hiper_plot (X, y, param_range, param_name):
    train_scores, test_scores = validation_curve(
                                    RandomForestClassifier(),
                                    X = X, y = y,
                                    param_name = param_name,
                                    param_range = param_range , cv = 3)

    # Calculate mean for training and test set scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")


    # Create plot
    plt.title("Validation Curve With Random Forest")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")



def plot_kmeans(kmeans_data, k, df_):

    original_labels = kmeans_data['kmeans_labels']
    y_grouped = kmeans_data['grouped_labels']
    centroids = kmeans_data['centroids']
    recall = kmeans_data['recall']
    fpr = kmeans_data['fpr']

    labels_df = pd.DataFrame({"KMEANS_LABELS": original_labels, "GROUPED_CLUSTERS": y_grouped})
    # plot comparison
    color_by_category(df_, labels_df, nrows=1, ncols=2, sup_title='K-Means over PCA, k = {}'.format(k), figsize=(16, 4))
    # last figure subplot handlers
    last_fig = plt.get_fignums()[-1]
    ax2 = plt.figure(last_fig).get_axes()[-2]
    for centroid in centroids:
        ax2.scatter(centroid[0], centroid[1], marker='x', c='black')

    ax2.get_legend().remove()
    # texts for 3rd subplot
    ax3 = plt.figure(last_fig).get_axes()[-1]
    ax3.set_title("Regrouped clusters")
    ax3.text(2, 10, s="Recall: {}\nfpr:{}".format(recall, fpr), fontsize=14, horizontalalignment='center')




def plot_kmeans_comparison(data, figsize=(8, 6), kmarkers = []):
    plt.figure(figsize=figsize)
    x = []
    recalls = []
    fprs = []
    for k in data.keys():
        x.append(k)
        recalls.append(data[k]['recall'])
        fprs.append(data[k]['fpr'])

    plt.plot(x, recalls, label="Recall", linestyle='--')
    plt.plot(x, fprs, label='FPR', linestyle='--')
    plt.title("k-means: Recall and fpr vs k")
    plt.xlabel("Number of clusters")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid()

    if len(kmarkers) > 0:
        # add markers to recall curve
        for marker in kmarkers:
            id = x.index(marker)
            print(marker, id)
            plt.scatter(x[id], recalls[id], marker='x', color='black')
            plt.text(x[id], recalls[id] + 0.015, horizontalalignment='center', fontsize=13, s="k = {}".format(marker))



def get_labels_mapping(labels_cluster1, n_clusters):
    # define labels that are to be assigned to same group (0 or 1)
    labels_1 = set(labels_cluster1)
    labels_0 = set([i for i in range(0, n_clusters)]) - labels_1
    mapping = {}
    for label in labels_1:
        mapping[label] = 1
    for label in labels_0:
        mapping[label] = 0
    return mapping


def get_recall_fpr(y, y_pred):
    recall = skm.recall_score(y, y_pred)
    tn, fp, fn, tp = skm.confusion_matrix(y, y_pred).ravel()
    fpr = fp / (fp + tn)
    return (round(recall, 2), round(fpr, 2))


def get_df_categorical_dummies(df):
    df = pd.concat([df, pd.get_dummies(df.GENERAL_OPINION, prefix='GENERAL_OPINION', drop_first=True)], axis=1)
    df.drop(['GENERAL_OPINION'], axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df.SPEAKER_POP, prefix='SPEAKER_POP', drop_first=True)], axis=1)
    df.drop(['SPEAKER_POP'], axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df.EVENT, prefix='EVENT', drop_first=True)], axis=1)
    df.drop(['EVENT'], axis=1, inplace=True)
    
    return df


def normalize_test_columns(df_train, df_test):
    missing_cols = set(df_train.columns) - set(df_test.columns)
    for c in missing_cols:
        df_test[c] = 0
    return df_test
