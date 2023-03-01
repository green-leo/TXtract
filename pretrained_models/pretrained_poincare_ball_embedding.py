from gensim.models.poincare import PoincareModel, PoincareRelations
from config import cfg


if __name__ == "__main__":
    # configures
    model_path = cfg.PRETRAINED_POINCARE_BALL_PATH
    file_path = cfg.CATE_HYPERNYM_FILE_PATH

    # model
    model = PoincareModel(PoincareRelations(file_path), size=cfg.CATE_DIM, negative=10)
    model.train(epochs=50)

    ### online training: update the vocabulary and continue training
    # online_relations = [('striped_skunk', 'mammal')]
    # model.build_vocab(online_relations, update=True)
    # model.train(epochs=50)

    model.save(model_path)

    # Test the model
    # print('26 vs 8:',model.kv.similarity('26','8'))
    # print('3831 vs 94:', model.kv.similarity('3831','94'))

    # print('Embedding for 8: ', model.kv['8'])


