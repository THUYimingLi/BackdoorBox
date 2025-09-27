import copy
import warnings

# from torchvision.transforms import v2
import numpy as np
import torch
import torch.nn as nn
import umap
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.cluster import HDBSCAN

from ..utils import test
from .base import Base

warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state. Use no seed for parallelism.")
warnings.filterwarnings("ignore", message="NumbaPendingDeprecationWarning.*")

class BatchNorm2d_ent(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_ent, self).__init__(num_features)
        # self.likehood = []
        self.max_log_prob = []
        self.min_log_prob = []

    def calculate_likelihood(self, activations, bn_mean, bn_var, epsilon=1e-5):

        bn_std = torch.sqrt(bn_var + epsilon)
        # Initialize a normal distribution with the BN parameters
        bn_mean = bn_mean.view(1, -1, 1, 1)
        bn_std = bn_std.view(1, -1, 1, 1)

        normal_dist = torch.distributions.Normal(bn_mean, bn_std)
        # [num_samples, num_chanels, H, W]
        # print(activations.size())
        with torch.no_grad():
            activations = normal_dist.log_prob(activations)
        activations = activations.view(activations.size(0), -1)
        likehood =  activations.min(dim=-1)[0]
        return likehood

    def forward(self, x):
        self.likehood = self.calculate_likelihood(x, self.running_mean.clone(), self.running_var.clone())
        return super(BatchNorm2d_ent, self).forward(x)

def replace_bn_with_ent(model):
    # model = copy.deepcopy(omodel)
    for name, module in model.named_children():
        # Check if the module is an instance of BatchNorm2d
        if isinstance(module, nn.BatchNorm2d):
            # Create a new instance of your custom BN layer
            new_bn = BatchNorm2d_ent(module.num_features)
            
            # Copy the parameters from the original BN layer to the new one
            new_bn.running_mean = module.running_mean.clone()
            new_bn.running_var = module.running_var.clone()
            new_bn.weight = nn.Parameter(module.weight.clone())
            new_bn.bias = nn.Parameter(module.bias.clone())
            
            # Replace the original BN layer with the new one
            setattr(model, name, new_bn)
        else:
            # Recursively apply the same operation to child modules
            replace_bn_with_ent(module)
    # return model
            
class FLARE(Base):
    """Identify and filter malicious training samples (FLARE: Towards Universal Dataset Purification against Backdoor Attacks. TIFS-2025).
    Args:
        xi (float): The hyper-parameter for the stability threshold.
    """

    def __init__(self, model, xi=0.02, seed=666, deterministic=False):
        super(FLARE, self).__init__(seed, deterministic)
        self.xi = xi
        self.model = model
        self.model.eval()
        
        torch.backends.cudnn.deterministic = True
    
    def test_acc(self, dataset, schedule):
        """Test curve model on test dataset

        Args:
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
        """
        model = self.model
        test(model, dataset, schedule)
    
    '''
    Calculate the likelihood of a batch training samples using a deep copy of the model with BatchNorm2d_ent batch normalization
    '''
    def cal_like(self, copym, imgs, wo_layer_num=1):
        batch_hoods = []
        copym(imgs)
        # cal the number of BN layers 
        layer_num = 0
        for (name1, module1) in copym.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                layer_num += 1
                
        indices = -1           
        for (name1, module1) in copym.named_modules():
            '''
            If the module is a BatchNorm2d layer, calculate the likelihood of the input images
            '''
            if isinstance(module1, torch.nn.BatchNorm2d):
                indices += 1
                if indices < layer_num-wo_layer_num:
                    '''Calculate the likelihood of the input images using the current BatchNorm2d layer'''
                    batch_hoods.append(module1.likehood)
        '''
        Stack the likelihoods of the input images from all BatchNorm2d layers into a tensor
        The shape of the batch_hoods tensor is [batch_size, num_BN_layers-wo_layer_num], where num_BN_layers is the total number of BatchNorm2d layers in the model, wo_layer_num is the number of BatchNorm2d layers to exclude from the likelihood calculation.
        '''
        batch_hoods = torch.stack(batch_hoods, dim=1) 
        # print(f'batch_hoods size: {batch_hoods.size()}')
        return batch_hoods      
    
    '''
    Computes the likelihood scores of training samples using a modified version of the current model  with BatchNorm2d_ent batch normalization

    This function creates a deep copy of the model, replaces its BatchNorm layers with BatchNorm2d_ent alternatives (via `replace_bn_with_ent`), and obtains the likelihood scores of the training data.

    Args:
        wo_layer_num (int): The number of BatchNorm layers to exclude from the likelihood calculation

    Returns:
        Tuple:
            - likelihoods (Tensor): A tensor of computed likelihood scores for all training samples.
            - ground_labels (ndarray): A NumPy array of the corresponding ground truth labels.
    '''
    def get_likehood(self, wo_layer_num): 
        copym = copy.deepcopy(self.model)
        replace_bn_with_ent(copym)
        copym.cuda()
        copym.eval()

        likelihoods = []
        y_true = []            
        # pred_labels = []
        with torch.no_grad():                     
            for idx, batch in enumerate(self.train_loader):
                img = batch[0].cuda()  # batch * channels * hight * width
                labels = batch[1]  # batch
                hoods = self.cal_like(copym, img, wo_layer_num=wo_layer_num)
                likelihoods.append(hoods)

        '''
        Stack the likelihoods of the input images from all BatchNorm2d layers into a tensor
        The shape of the likelihoods tensor is [num_training_ samples, num_BN_layers-wo_layer_num], where num_BN_layers is the total number of BatchNorm2d layers in the model, wo_layer_num is the number of BatchNorm2d layers to exclude from the likelihood calculation.
        '''
        likelihoods = torch.cat(likelihoods, dim=0).cpu()
        # print(f'likelihoods.size(): {likelihoods.size()}')
        # torch.save(likelihoods, hood_path)
        
        return likelihoods
    
    '''
    Identify the poisoned samples from the poisoned training set        
    Args:   
        poisoned_trainset (torch.utils.data.Dataset): The poisoned training dataset.
        y_true (np.ndarray): Ground-truth poison status of the training samples, 1 for poisoned and 0 for clean.
    '''
    def detect(self, poisoned_trainset, y_true):
        self.train_loader = torch.utils.data.DataLoader(
                poisoned_trainset,
                batch_size=128, shuffle=False)
        self.trainset = poisoned_trainset
        
        bn_num = 0 
        for (name1, module1) in self.model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                bn_num += 1
        '''
        depths is the number of layers to traverse down the condensed tree to find the first cluster with a significant drop in lambda value;
        depths = [3] means that the algorithm will traverse down 3 layers in the condensed tree.
        dp is the value of "d" in the original paper (line 9 in Algorithm 1 on page 8)
        '''
        depths = [3]
        for dp in depths:
            print(f'**************************depth: {dp}*****************************************')
            '''
           The goal is to find the status (the optimal wo_layer_num) that all benign samples are close to each other and far away from the poisoned samples
            
            wo_layer_num is the number of BatchNorm2d layers to exclude from the likelihood calculation.
            For example, if wo_layer_num = 0, it means that all BatchNorm2d layers are included in the likelihood calculation;
            if wo_layer_num = 1, it means that the last BatchNorm2d layer is excluded from the likelihood calculation.

            we progressively exclude the last BatchNorm2d layer from the likelihood calculation;

            '''
            for wo_layer_num in range(0, bn_num):
                print(f'======================= wo_layer_num: {wo_layer_num} =======================')
                '''Get the likelihoods of the training samples using the modified model with BatchNorm2d_ent batch normalization'''
                likelihoods = self.get_likehood(wo_layer_num)
                '''Perform UMAP dimensionality reduction on the likelihoods'''
                umap_model = umap.UMAP(n_neighbors=40, min_dist=0, n_components=2, random_state=42)
                X = umap_model.fit_transform(likelihoods.cpu().numpy())
                '''HDBSCAN clustering on the UMAP-reduced data'''
                clusterer = HDBSCAN(min_cluster_size=100, gen_min_span_tree=True, prediction_data=True)
                clusterer.fit(X)
                '''Get the condensed tree and linkage tree from the clustering results'''
                ori_tree = clusterer.condensed_tree_
                tree = clusterer.condensed_tree_.to_pandas()
                link_tree = clusterer.single_linkage_tree_

                '''
                The condensed tree is a DataFrame with the following columns:
                - parent: The parent node ID.   
                - child: The child node ID.
                - lambda_val: The density level of the child node.
                - child_size: The size of the child node (number of samples in the cluster).
                - ...: Other columns may include additional information about the clustering process.

                For example, the condensed tree may look like this:
                | parent | child | lambda\_val | child\_size | ... |
                | ------ | ----- | ----------- | ----------- | --- |
                | 0      | 3     | 0.05        | 42          | ... |
                | 0      | 4     | 0.05        | 28          | ... |
                | 3      | 5     | 0.12        | 10          | ... |

                Interpretation:

                    Cluster 0 is a root cluster that splits into clusters 3 and 4 at lambda = 0.05.

                    Cluster 3 later splits again into cluster 5 at lambda = 0.12.

                    Therefore:

                    Cluster 3 is a child of cluster 0.

                    Cluster 0 is a parent of clusters 3 and 4.

                    Cluster 5 is a child of cluster 3.

                '''


                '''visualize the clustering results in condensed tree style'''           
                # tree_path = f'{wo_save_root}/condensed_tree.pdf'
                # plt.rcParams['font.family'] = 'Times New Roman'
                # plt.figure(figsize=(15, 12))
                # if not os.path.exists(tree_path):
                #     clusterer.condensed_tree_.plot()
                #     plt.savefig(tree_path)
                '''Get the node with the maximum child size'''
                max_node = tree.loc[tree['child_size'].idxmax()]
                '''the node with the maximum child size is the root node of the condensed tree'''
                # node_id = max_node['child']
                node_id = max_node['child']
                

                '''
                Find the first cluster with a significant drop in lambda value;
                A significant drop is defined as the difference in lambda value exceeding a given threshold (xi).
                A significant increase in lambda values typically indicates that the resulting clusters are more stable and well-separated.
                '''
                def find_first_large_drop(tree, node_id, xi=0.02, depth=3):
                    """
                    Starting from a given node, traverse downward through the condensed tree to find the first cluster 
                    where the lambda value drops significantly — that is, the difference in lambda exceeds a given threshold. 
                    This point is considered a potential cluster split.

                    :param tree: A DataFrame representation of the condensed tree, where each row represents a clustering step.
                    :param node_id: The starting node ID from which to begin the traversal.
                    :param  stability threshold xi: The threshold for the difference in lambda to determine a significant drop.
                    :return: Information about the subtree (cluster) that satisfies the condition.
                    """
                    sub_trees_info = []
                    stack = [node_id] # stack is the "Q" in the original paper (line 6 in Algorithm 1 on page 8 )
                    iterations = 0
                    # max_iterations = 3

                    '''depths is the number of layers to traverse down the condensed tree to find the first cluster with a significant drop in lambda value
                    If the depth is set to 3, it means that the algorithm will traverse down 3 layers in the condensed tree.'''
                    max_iterations = depth
                    while stack and iterations < max_iterations:
                        current_id = stack.pop()
                        '''
                        Get the current data from the tree DataFrame using the current child id
                        for example, if current_id is 3, it will get the row where child is 3:
                            | 0      | 3     | 0.05        | 42          | ... |
                        This row contains the parent id, child id, lambda value, and max child size of the current tree.
                        
                        We restrict the traversal to the subtree with the largest child size, which is assumed to represent the benign cluster, since poisoned samples are rare and occupy only a small fraction of the dataset.
                        
                        '''
                        current_node = tree[tree['child'] == current_id].iloc[0]

                        # lambda_val is the density level of current_node 
                        lambda_val = current_node['lambda_val']
                        print(f'current_node:\n {current_node}')

                        '''Get all the children of the current node'''
                        children = tree.loc[tree['parent'] == current_id]
                        max_child = tree.loc[children['child_size'].idxmax()]
                        print(f"max child:\n {max_child}")
                        # max_child = tree.loc[children['child_size'].idxmax()]
                        # cal the  cluster stability
                        lambda_diff = max_child['lambda_val'] - lambda_val
                        print(f'lambda_diff: {lambda_diff}')
                        
                        '''Given that benign samples are heterogeneous and cover various classes, the stability of the benign cluster indicates that the BN layers identified along this subtree are the appropriate ones to be excluded.'''
                        if lambda_diff > xi:
                            return max_child, lambda_diff
                        
                        stack.append(max_child['child'])
                        iterations += 1

                    return None, None
                
                
                split_child, lambda_diff = find_first_large_drop(tree, node_id, xi=self.xi, depth=dp)

                print(f'**********************find the best wo_layer_num: {wo_layer_num}**********************')
                thres_lambda_val = split_child['lambda_val'] - 0.0001
                labels = link_tree.get_clusters(1/thres_lambda_val, min_cluster_size=100)
                final_labels = np.zeros(len(labels), dtype=int)
                # 计算每个聚类的大小并找出最大的聚类
                unique_labels, counts = np.unique(labels, return_counts=True)
                max_cluster_label = unique_labels[np.argmax(counts)]
                for label in unique_labels:
                    if label == -1:  # 噪音点已经设置为0
                        continue
                    elif label == max_cluster_label:
                        final_labels[labels == label] = 0
                    else:
                        final_labels[labels == label] = 1
                        
                y_preds = final_labels
                pred_clean_indices = torch.where(torch.from_numpy(y_preds) == 0)[0]
                # torch.save(clean_indices, f'{name_root}/pred_clean_indices.pt')
        
                print(f'y_preds: {y_preds}')
                print(f'y_true: {y_true}')
                tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_preds).ravel()
                # tn, fp, fn, tp = metrics.confusion_matrix(y_true, labels).ravel()
                print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
                print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
                print("FNR: {:.2f}".format(fn / (tp + fn) * 100))        
                break
        
    # train model on the remain benign samples
    # def train_on_filter(self):
    #     args = self.args
    #     # indices_path = f"{supervisor.get_poison_set_dir(args)}/only_min/pred_indices/clean_model.pt"
    #     name_root = f'{supervisor.get_poison_set_dir(args)}/FLARE/'
    #     indices_path = f'{name_root}/clean_indices.pt'
    #     clean_indices = torch.load(indices_path).tolist()
    #     clean_trainset = Subset(self.trainset, clean_indices)
    #     self.train_model(self.args, clean_indices, model_path)
    
    
   