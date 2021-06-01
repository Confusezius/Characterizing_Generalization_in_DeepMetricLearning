import sys
sys.path.insert(0, '..')
import copy

import faiss
import numpy as np
import time
import torch
from tqdm import tqdm

from metrics import e_recall, a_recall, dists, rho_spectrum
from metrics import nmi, f1, mAP, mAP_c, mAP_1000
from utilities import misc


def select(metricname, opt=None):
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)
    elif 'a_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return a_recall.Metric(k)
    elif metricname=='nmi':
        return nmi.Metric()
    elif metricname=='mAP_c':
        return mAP_c.Metric()
    elif metricname=='mAP_1000':
        return mAP_1000.Metric()
    elif metricname=='mAP':
        return mAP.Metric()
    elif metricname=='f1':
        return f1.Metric()
    elif 'dists' in metricname:
        mode = metricname.split('@')[-1]
        return dists.Metric(mode)
    elif 'rho_spectrum' in metricname:
        mode = int(metricname.split('@')[-1])
        embed_dim = 128 if opt is None else opt.embed_dim
        return rho_spectrum.Metric(embed_dim, mode=mode, opt=opt)
    else:
        raise NotImplementedError("Metric {} not available!".format(metricname))


class MetricComputer():
    def __init__(self, metric_names, opt):
        self.pars            = opt
        self.metric_names    = metric_names
        self.list_of_metrics = [select(metricname, opt) for metricname in metric_names]
        self.requires        = [metric.requires for metric in self.list_of_metrics]
        self.requires        = list(set([x for y in self.requires for x in y]))

    def update_test_data(self):
        pass


    def compute_standard(self, opt, model, dataloader, evaltypes, device, mode, **kwargs):
        evaltypes = copy.deepcopy(evaltypes)

        n_classes = opt.n_classes
        if opt.new_test_dataset != 'None' and mode == 'Test':
            n_classes = opt.n_test_classes
        ###
        feature_colls  = {key:[] for key in evaltypes}

        ###
        _ = model.eval()
        with torch.no_grad():
            target_labels = []
            final_iter = tqdm(dataloader, desc='Embedding Data...'.format(len(evaltypes)))
            image_paths= [x[0] for x in dataloader.dataset.image_list]
            for idx,inp in enumerate(final_iter):
                input_img,target = inp[1], inp[0]
                target_labels.extend(target.numpy().tolist())

                out_dict     = model(input_img.to(device))
                if 'multifeature' in model.name:
                    out          = out_dict['embeds']
                    for evaltype in evaltypes:
                        weights      = [float(x) for x in evaltype.split('-')[1:]]
                        subevaltypes = evaltype.split('Combined_')[-1].split('-')[0].split('_')
                        weighted_subfeatures = [weights[i]*out[subevaltype] for i,subevaltype in enumerate(subevaltypes)]
                        if 'normalize' in model.name:
                            feature_colls[evaltype].extend(torch.nn.functional.normalize(torch.cat(weighted_subfeatures, dim=-1), dim=-1).cpu().detach().numpy().tolist())
                        else:
                            feature_colls[evaltype].extend(torch.cat(weighted_subfeatures, dim=-1).cpu().detach().numpy().tolist())
                else:
                    out      = out_dict['embeds']
                    feature_colls['embeds'].extend(out.cpu().detach().numpy().tolist())

            target_labels = np.hstack(target_labels).reshape(-1,1)

        ###
        computed_metrics = {evaltype:{} for evaltype in evaltypes}
        extra_infos      = {evaltype:{} for evaltype in evaltypes}


        ###
        faiss.omp_set_num_threads(self.pars.kernels)
        # faiss.omp_set_num_threads(self.pars.kernels)
        res = None
        torch.cuda.empty_cache()
        if self.pars.evaluate_on_gpu:
            res = faiss.StandardGpuResources()

        for evaltype in evaltypes:
            features = np.vstack(feature_colls[evaltype]).astype('float32')

            start = time.time()
            if 'kmeans' in self.requires:
                ### Set CPU Cluster index
                cluster_index = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: cluster_index = faiss.index_cpu_to_gpu(res, 0, cluster_index)
                kmeans = faiss.Clustering(features.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                ### Train Kmeans
                kmeans.train(features, cluster_index)
                centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])


            if 'kmeans_nearest' in self.requires:
                faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids)
                _, computed_cluster_labels = faiss_search_index.search(features, 1)

            if 'nearest_features' in self.requires:
                faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(features)

                max_kval            = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
                _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))
                k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

            ###
            if self.pars.evaluate_on_gpu:
                features = torch.from_numpy(features).to(self.pars.device)

            start = time.time()
            for metric in self.list_of_metrics:
                input_dict = {}
                if 'features' in metric.requires:         input_dict['features'] = features
                if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels
                if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
                if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
                if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes
                computed_metrics[evaltype][metric.name] = metric(**input_dict)

            extra_infos[evaltype] = {'features':features, 'target_labels':target_labels,
                                     'image_paths': dataloader.dataset.image_paths,
                                     'query_image_paths':None, 'gallery_image_paths':None}

        torch.cuda.empty_cache()

        return computed_metrics, extra_infos




        def compute_query_gallery(self, opt, model, query_dataloader, gallery_dataloader, evaltypes, device, **kwargs):
            n_classes = opt.n_classes
            query_image_paths   = np.array([x[0] for x in query_dataloader.dataset.image_list])
            gallery_image_paths = np.array([x[0] for x in gallery_dataloader.dataset.image_list])
            _ = model.eval()

            ###
            query_feature_colls   = {evaltype:[] for evaltype in evaltypes}
            gallery_feature_colls = {evaltype:[] for evaltype in evaltypes}

            ### For all test images, extract features
            with torch.no_grad():
                ### Compute Query Embedding Features
                query_target_labels = []
                query_iter = tqdm(query_dataloader, desc='Extraction Query Features')
                for idx,inp in enumerate(query_iter):
                    input_img,target = inp[1], inp[0]
                    query_target_labels.extend(target.numpy().tolist())
                    out = model(input_img.to(device))
                    if isinstance(out, tuple): out, aux_f = out


                    ### Include Metrics for separate linear layers.
                    if hasattr(model, 'merge_linear'):
                        merged_features = model.merge_linear(torch.cat([feat for feat in out.values()], dim=-1))
                        if 'merged_discriminative' not in query_feature_colls: query_feature_colls['merged_discriminative']   = []
                        query_feature_colls['merged_discriminative'].extend(merged_features.cpu().detach().numpy().tolist())
                    if hasattr(model, 'separate_linear'):
                        sep_features    = model.separate_linear(aux_f)
                        if 'separate_discriminative' not in query_feature_colls: query_feature_colls['separate_discriminative'] = []
                        query_feature_colls['separate_discriminative'].extend(merged_features.cpu().detach().numpy().tolist())


                    for evaltype in evaltypes:
                        if 'Combined' not in evaltype:
                            if isinstance(out, dict):
                                query_feature_colls[evaltype].extend(out[evaltype].cpu().detach().numpy().tolist())
                            else:
                                query_feature_colls[evaltype].extend(out.cpu().detach().numpy().tolist())

                    for evaltype in evaltypes:
                        if 'Combined' in evaltype:
                            weights      = [float(x) for x in evaltype.split('-')[1:]]
                            subevaltypes = evaltype.split('Combined_')[-1].split('-')[0].split('_')
                            weighted_subfeatures = [weights[i]*out[subevaltype] for i,subevaltype in subevaltypes]
                            query_feature_colls[evaltype].extend(torch.nn.functional.normalize(torch.cat(weighted_subfeatures, dim=-1), dim=-1).cpu().detach().numpy().tolist())



                ### Compute Gallery Embedding Features
                gallery_target_labels = []
                gallery_iter = tqdm(gallery_dataloader, desc='Extraction Gallery Features')
                for idx,inp in enumerate(gallery_iter):
                    input_img,target = inp[1], inp[0]
                    gallery_target_labels.extend(target.numpy().tolist())
                    out = model(input_img.to(device))
                    if isinstance(out, tuple): out, aux_f = out


                    ### Include Metrics for separate linear layers.
                    if hasattr(model, 'merge_linear'):
                        merged_features = model.merge_linear(torch.cat([feat for feat in out.values()], dim=-1))
                        if 'merged_discriminative' not in gallery_feature_colls: gallery_feature_colls['merged_discriminative']   = []
                        gallery_feature_colls['merged_discriminative'].extend(merged_features.cpu().detach().numpy().tolist())
                    if hasattr(model, 'separate_linear'):
                        sep_features    = model.separate_linear(aux_f)
                        if 'separate_discriminative' not in gallery_feature_colls: gallery_feature_colls['separate_discriminative'] = []
                        gallery_feature_colls['separate_discriminative'].extend(merged_features.cpu().detach().numpy().tolist())


                    for evaltype in evaltypes:
                        if 'Combined' not in evaltype:
                            if isinstance(out, dict):
                                gallery_feature_colls[evaltype].extend(out[evaltype].cpu().detach().numpy().tolist())
                            else:
                                gallery_feature_colls[evaltype].extend(out.cpu().detach().numpy().tolist())

                    for evaltype in evaltypes:
                        if 'Combined' in evaltype:
                            weights      = [float(x) for x in evaltype.split('-')[1:]]
                            subevaltypes = evaltype.split('Combined_')[-1].split('-')[0].split('_')
                            weighted_subfeatures = [weights[i]*out[subevaltype] for i,subevaltype in subevaltypes]
                            gallery_feature_colls[evaltype].extend(torch.nn.functional.normalize(torch.cat(weighted_subfeatures, dim=-1), dim=-1).cpu().detach().numpy().tolist())


                ###
                query_target_labels, gallery_target_labels  = np.hstack(query_target_labels).reshape(-1,1), np.hstack(gallery_target_labels).reshape(-1,1)
                computed_metrics = {evaltype:{} for evaltype in evaltypes}
                extra_infos      = {evaltype:{} for evaltype in evaltypes}

                if hasattr(model, 'merge_linear'):    evaltypes.append('merged_discriminative')
                if hasattr(model, 'separate_linear'): evaltypes.append('separate_discriminative')

                ###
                for evaltype in evaltypes:
                    query_features   = np.vstack(query_feature_colls[evaltype]).astype('float32')
                    gallery_features = np.vstack(gallery_feature_colls[evaltype]).astype('float32')

                    if 'kmeans' in self.requires:
                        ### Set CPU Cluster index
                        stackset    = np.concatenate([query_features, gallery_features],axis=0)
                        stacklabels = np.concatenate([query_target_labels, gallery_target_labels],axis=0)
                        cluster_index = faiss.IndexFlatL2(stackset.shape[-1])
                        kmeans            = faiss.Clustering(stackset.shape[-1], n_classes)
                        kmeans.niter = 20
                        kmeans.min_points_per_centroid = 1
                        kmeans.max_points_per_centroid = 1000000000
                        ### Train Kmeans
                        kmeans.train(stackset, cluster_index)
                        centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, stackset.shape[-1])

                    if 'kmeans_nearest' in self.requires:
                        faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
                        faiss_search_index.add(centroids)
                        _, computed_cluster_labels = faiss_search_index.search(stackset, 1)

                    if 'nearest_features' in self.requires:
                        faiss_search_index  = faiss.IndexFlatL2(gallery_features.shape[-1])
                        faiss_search_index.add(gallery_features)
                        _, k_closest_points = faiss_search_index.search(query_features, int(np.max(k_vals)))
                        k_closest_classes   = gallery_target_labels.reshape(-1)[k_closest_points]

                    ###
                    for metric in self.list_of_metrics:
                        input_dict = {}
                        if 'features' in metric.requires:         input_dict['features'] = features
                        if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels
                        if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
                        if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
                        if 'nearest_features' in metric.requires: input_dict['k_closest_classes']

                        computed_metrics[evaltype][metric.name] = metric(**input_dict)

                    ###
                    extra_infos[evaltype] = {'features':features, 'image_paths': None, 'target_labels': target_labels,
                                             'query_image_paths':  query_dataloader.dataset.image_paths,
                                             'gallery_image_paths':gallery_dataloader.dataset.image_paths}

                return computed_metrics, extra_info





class NonRedundantMetricComputer():
    def __init__(self, metric_names):
        self.metric_names    = metric_names
        self.list_of_metrics = [select(metricname) for metricname in metric_names]
        self.requires        = [metric.requires for metric in self.list_of_metrics]
        self.requires        = list(set([x for y in self.requires for x in y]))

    def compute_standard(self, features, labels, device,
                         num_clusters, kernels=6, evaluate_on_gpu=False):
        start = time.time()
        computed_metrics = {}

        ###
        faiss.omp_set_num_threads(kernels)
        res = None
        torch.cuda.empty_cache()
        if evaluate_on_gpu:
            res = faiss.StandardGpuResources()

        features = features.astype('float32')

        ###
        if 'kmeans' in self.requires:
            ### Set CPU Cluster index
            cluster_index = faiss.IndexFlatL2(features.shape[-1])
            if res is not None: cluster_index = faiss.index_cpu_to_gpu(res, 0, cluster_index)
            kmeans = faiss.Clustering(features.shape[-1], num_clusters)
            kmeans.niter = 20
            kmeans.min_points_per_centroid = 1
            kmeans.max_points_per_centroid = 1000000000
            ### Train Kmeans
            kmeans.train(features, cluster_index)
            centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])

        if 'kmeans_nearest' in self.requires:
            faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
            if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
            faiss_search_index.add(centroids)
            _, computed_cluster_labels = faiss_search_index.search(features, 1)

        if 'nearest_features' in self.requires:
            faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
            if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
            faiss_search_index.add(features)

            max_kval            = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
            _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))
            k_closest_classes   = labels.reshape(-1)[k_closest_points[:,1:]]

        ###
        if evaluate_on_gpu:
            features = torch.from_numpy(features).to(self.pars.device)

        start = time.time()
        for metric in self.list_of_metrics:
            input_dict = {}
            if 'features' in metric.requires:         input_dict['features'] = features
            if 'target_labels' in metric.requires:    input_dict['target_labels'] = labels
            if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
            if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
            if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes
            computed_metrics[metric.name] = metric(**input_dict)

        return computed_metrics
