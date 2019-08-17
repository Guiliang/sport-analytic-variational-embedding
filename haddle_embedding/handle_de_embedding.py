import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.manifold import TSNE
from config.de_embed_config import DEEmbedCongfig
from support.model_tools import get_model_and_log_name


def combine_de_embeddings():
    combined_embeddings = None
    predicted_target_all = ['state', 'action', 'reward']
    for predicted_target in predicted_target_all:
        de_config_path = "../ice_hockey_{0}_de.yaml".format(predicted_target)
        de_config = DEEmbedCongfig.load(de_config_path)
        save_network_dir, log_dir = get_model_and_log_name(config=de_config, model_catagoery='de_embed',
                                                           train_flag=False)
        save_embed_dir = save_network_dir.replace('de_embed_saved_networks', 'store_embedding'). \
            replace('de_model_saved_NN', 'de_model_save_embedding')

        with open(save_embed_dir + '/embedding_matrix_game{0}.csv'.format(str(901)), 'r') as f:
            csv_reader = csv.reader(f)
            read_embedding = []
            for row in csv_reader:
                # print(row)
                read_embedding.append(row)
        if combined_embeddings is None:
            combined_embeddings = np.asarray(read_embedding)
        else:
            combined_embeddings = np.concatenate([combined_embeddings, read_embedding], axis=1)

    return combined_embeddings


def classify_players(embeddings, cluster_name='ap'):
    if cluster_name == 'km':
        cluster = KMeans(n_clusters=5).fit(embeddings)
        print 'The number of cluster is {0}'.format(len(cluster.cluster_centers_))
    elif cluster_name == 'ap':
        cluster = AffinityPropagation(damping=0.8).fit(embeddings)
        print 'The number of cluster is {0}'.format(len(cluster.cluster_centers_indices_))
    else:
        raise ValueError()
    cluster_number = cluster.predict(embeddings)

    return cluster_number


def dimensional_reduction(embeddings):
    dr_embedding = TSNE(n_components=2).fit_transform(embeddings)
    print 'finish t-sne'
    return dr_embedding


def visualize_embeddings(data, cluster_number):
    num_cluster = np.max(cluster_number, axis=0)
    for cluster in range(0, num_cluster):
        indices = [i for i, x in enumerate(cluster_number) if x == cluster]
        plt.scatter(data[indices, 0], data[indices, 1], s=50)
    # plt.show()
    plt.savefig('./player_de_cluster{0}.png'.format(str(num_cluster)))


if __name__ == '__main__':
    combined_embeddings = combine_de_embeddings()
    dr_embedding = dimensional_reduction(embeddings=combined_embeddings)
    cluster_number = classify_players(embeddings=dr_embedding)
    visualize_embeddings(data=dr_embedding, cluster_number=cluster_number)
