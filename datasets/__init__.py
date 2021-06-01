import datasets.cub200
import datasets.cars196
import datasets.stanford_online_products


def select(dataset, opt, data_path, splitpath=None):
    if splitpath is None:
        if 'cub200' in dataset:
            return cub200.DefaultDatasets(opt, data_path)
        if 'cars196' in dataset:
            return cars196.DefaultDatasets(opt, data_path)
        if 'online_products' in dataset:
            return stanford_online_products.DefaultDatasets(opt, data_path)
    else:
        if 'cub200' in dataset:
            return cub200.OODatasets(opt, data_path, splitpath)
        if 'cars196' in dataset:
            return cars196.OODatasets(opt, data_path, splitpath)
        if 'online_products' in dataset:
            return stanford_online_products.OODatasets(opt, data_path, splitpath)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196 & stanford_online_products!'.format(dataset))
