import numpy as np
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice


class HybridLDASplitter(BaseSplitter):
    """
    This splitter combines LDA (Dirichlet distribution) with data sampling from 
    another dataset for specific clients.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha 
            generates more extreme heterogeneous scenario see 
            ``np.random.dirichlet``
        special_client_indices (list, optional): List of client indices that 
            should sample from another dataset. If None, will use percentage.
        special_client_percentage (float, optional): Percentage of clients 
            that should sample from another dataset (0.0 to 1.0). 
            Used only if special_client_indices is None.
        secondary_dataset: Another dataset to sample from for special clients
        secondary_dataset_ratio (float): Ratio of data to sample from 
            secondary dataset (0.0 to 1.0)
    """
    
    def __init__(self, client_num, alpha=0.1, special_client_indices=None, 
                 special_client_percentage=None, secondary_dataset=None, 
                 secondary_dataset_ratio=0.5, save_plot=False, exp_name="experiment", 
                 plot_dir="./0_Results"):
        self.alpha = alpha
        self.secondary_dataset_ratio = secondary_dataset_ratio
        self.save_plot = save_plot
        self.exp_name = exp_name
        self.plot_dir = plot_dir
        
        # Store secondary dataset reference - will be set when dataset is loaded
        self.secondary_dataset = secondary_dataset
        
        # Determine which clients are special
        if special_client_indices is not None:
            self.special_client_indices = special_client_indices
        elif special_client_percentage is not None:
            num_special = max(1, int(client_num * special_client_percentage))
            self.special_client_indices = np.random.choice(
                client_num, size=num_special, replace=False
            ).tolist()
        else:
            # Default: 20% of clients are special
            num_special = max(1, int(client_num * 0.2))
            self.special_client_indices = np.random.choice(
                client_num, size=num_special, replace=False
            ).tolist()
        
        super(HybridLDASplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset
        
        # Convert dataset to list format
        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            if 'categories' in tmp_dataset[0]:
                label = np.array([x['categories'] for x in tmp_dataset])
            elif 'label' in tmp_dataset[0]:
                label = np.array([x['label'] for x in tmp_dataset])
            else:
                raise ValueError('Cannot find label or categories in dataset')
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')
        
        # Get LDA-based split for all clients
        idx_slice = dirichlet_distribution_noniid_slice(
            label, self.client_num, self.alpha, prior=prior
        )
        
        # For special clients, replace some of their data with samples from secondary dataset
        # Get secondary dataset from dataset object if available
        secondary_dataset = getattr(dataset, 'secondary_dataset', self.secondary_dataset)
        if secondary_dataset is not None:
            idx_slice = self._replace_data_for_special_clients(
                dataset, idx_slice, label, secondary_dataset
            )
        
        # Convert to final format
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        
        # Generate visualization if requested and this is the training data (not test/val)
        # Check if this is training data by looking at the dataset size relative to expected total
        if self.save_plot and len(dataset) > 1000:  # Only visualize if dataset is large enough (training data)
            self._generate_visualization(dataset, idx_slice)
        
        return data_list

    def _replace_data_for_special_clients(self, dataset, idx_slice, label, secondary_dataset):
        """
        Replace data for special clients by sampling from secondary dataset.
        """
        # Check if secondary dataset is available
        if not secondary_dataset or len(secondary_dataset) == 0:
            print("Warning: Secondary dataset is empty or not loaded. Skipping data replacement.")
            return idx_slice
        
        # Convert secondary dataset to list format
        tmp_secondary = [ds for ds in secondary_dataset]
        if len(tmp_secondary) == 0:
            print("Warning: Secondary dataset is empty. Skipping data replacement.")
            return idx_slice
            
        if isinstance(tmp_secondary[0], tuple):
            secondary_label = np.array([y for x, y in tmp_secondary])
        elif isinstance(tmp_secondary[0], dict):
            if 'categories' in tmp_secondary[0]:
                secondary_label = np.array([x['categories'] for x in tmp_secondary])
            elif 'label' in tmp_secondary[0]:
                secondary_label = np.array([x['label'] for x in tmp_secondary])
            else:
                raise ValueError('Cannot find label or categories in secondary dataset')
        else:
            raise TypeError(f'Unsupported secondary dataset format {type(tmp_secondary[0])}')
        
        # Get unique labels from both datasets
        primary_labels = np.unique(label)
        secondary_labels = np.unique(secondary_label)
        common_labels = np.intersect1d(primary_labels, secondary_labels)
        
        if len(common_labels) == 0:
            raise ValueError("No common labels found between primary and secondary datasets")
        
        # Create mapping from secondary dataset indices to primary dataset indices
        secondary_to_primary_map = {}
        for label_val in common_labels:
            primary_indices = np.where(label == label_val)[0]
            secondary_indices = np.where(secondary_label == label_val)[0]
            
            # Create a mapping by sampling with replacement
            if len(secondary_indices) > 0 and len(primary_indices) > 0:
                # Sample secondary indices to map to primary indices
                sampled_secondary = np.random.choice(
                    secondary_indices, 
                    size=len(primary_indices), 
                    replace=True
                )
                secondary_to_primary_map.update(
                    dict(zip(sampled_secondary, primary_indices))
                )
        
        # Replace data for special clients
        modified_idx_slice = [idx_list.copy() for idx_list in idx_slice]
        
        for client_idx in self.special_client_indices:
            if client_idx < len(modified_idx_slice):
                original_indices = modified_idx_slice[client_idx]
                num_to_replace = int(len(original_indices) * self.secondary_dataset_ratio)
                
                if num_to_replace > 0 and len(secondary_to_primary_map) > 0:
                    # Get indices to replace (randomly selected)
                    indices_to_replace = np.random.choice(
                        len(original_indices), 
                        size=min(num_to_replace, len(original_indices)), 
                        replace=False
                    )
                    
                    # Sample from secondary dataset
                    secondary_indices = list(secondary_to_primary_map.keys())
                    if len(secondary_indices) > 0:
                        sampled_secondary = np.random.choice(
                            secondary_indices, 
                            size=len(indices_to_replace), 
                            replace=True
                        )
                        
                        # Replace the selected indices
                        for i, replace_idx in enumerate(indices_to_replace):
                            if i < len(sampled_secondary):
                                # Map secondary dataset index to primary dataset index
                                mapped_idx = secondary_to_primary_map[sampled_secondary[i]]
                                original_indices[replace_idx] = mapped_idx
                        
                        # Shuffle the modified indices
                        np.random.shuffle(original_indices)
        
        return modified_idx_slice

    def _generate_visualization(self, dataset, idx_slice):
        """Generate visualization plots for the data partition."""
        try:
            from federatedscope.core.splitters.visualization import DataPartitionVisualizer
            
            visualizer = DataPartitionVisualizer(save_dir=self.plot_dir)
            
            # Create main partition plot with special client highlighting
            plot_path = visualizer.plot_data_partition(
                dataset=dataset,
                indices_list=idx_slice,
                splitter_name="hybrid_lda",
                exp_name=self.exp_name,
                special_clients=self.special_client_indices
            )
            
            # Create hybrid-specific analysis plot
            hybrid_analysis_path = visualizer.plot_hybrid_splitter_analysis(
                dataset=dataset,
                indices_list=idx_slice,
                special_clients=self.special_client_indices,
                secondary_dataset_ratio=self.secondary_dataset_ratio,
                exp_name=self.exp_name
            )
            
            # Create summary plot
            summary_path = visualizer.create_summary_plot(
                dataset=dataset,
                indices_list=idx_slice,
                splitter_name="hybrid_lda",
                exp_name=self.exp_name,
                special_clients=self.special_client_indices
            )
            
            print(f"Hybrid LDA Splitter visualization saved:")
            print(f"  - Partition plot: {plot_path}")
            print(f"  - Hybrid analysis: {hybrid_analysis_path}")
            print(f"  - Summary plot: {summary_path}")
            
        except ImportError as e:
            print(f"Warning: Could not generate visualization plots: {e}")
        except Exception as e:
            print(f"Warning: Error generating visualization plots: {e}")

    def get_special_client_info(self):
        """
        Returns information about which clients are special.
        """
        return {
            'special_client_indices': self.special_client_indices,
            'num_special_clients': len(self.special_client_indices),
            'special_client_percentage': len(self.special_client_indices) / self.client_num
        }
