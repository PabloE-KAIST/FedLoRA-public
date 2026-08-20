"""Configuration file for composition of different aggregators, messages,
handlers, etc.

    - The method `local` indicates that the clients only locally
    train their model without sharing any training related information
    - The method `global` indicates that the only one client locally trains
    using all data

"""

AGGREGATOR_TYPE = {
    "local": "no_communication",  # the clients locally train their model
    # without sharing any training related info
    "global": "no_communication",  # only one client locally train all data,
    # i.e., totally global training
    "fedavg": "clients_avg",  # FedAvg
    "pfedme": "server_clients_interpolation",  # pFedMe,  + server-clients
    # interpolation
    "ditto": "clients_avg",  # Ditto
    "fedsageplus": "clients_avg",
    "gcflplus": "clients_avg",
    "fedgc": "clients_avg",
    "fedopt": "fedopt",
    "heterolora": "heterolora",  # HeteroLoRA for heterogeneous LoRA ranks
    "hetlora": "hetlora",  # HetLoRA Complete: rank self-pruning + sparsity-weighted aggregation
    "adasparse_lora": "adasparse_lora",  # AdaSparse-LoRA: index-aware per-component aggregation
    "adasparse_lorav2": "adasparse_lorav2",  # AdaSparse-LoRAv2: Stage 1 structural pruning + Stage 2 communication sparsity
    "adasparse_lorav3": "adasparse_lorav3"  # AdaSparse-LoRAv3: True layer-aware component identity + cross-layer competition
}

CLIENTS_TYPE = {
    "local": "normal",
    "fedavg": "normal",  # FedAvg
    "pfedme": "normal_loss_regular",  # pFedMe, + regularization-based local
    # loss
    "ditto": "normal",  # Ditto, + local training for distinct personalized
    # models
    "fedsageplus": "fedsageplus",  # FedSage+ for graph data
    "gcflplus": "gcflplus",  # GCFL+ for graph data
    "fedgc": "fedgc",
    "gradascent": "gradascent"
}

SERVER_TYPE = {
    "local": "normal",
    "fedavg": "normal",  # FedAvg
    "pfedme": "normal",  # pFedMe, + regularization-based local loss
    "ditto": "normal",  # Ditto, + local training for distinct personalized
    # models
    "fedsageplus": "fedsageplus",  # FedSage+ for graph data
    "gcflplus": "gcflplus",  # GCFL+ for graph data
    "fedgc": "fedgc"
}