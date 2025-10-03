import torch

"""
Saves Model
params: 
    model: the model to be saved
    model_args: dictionary, the arguments necessary to create the model
        keys:
            "num_node_features": number of input features per node in graph
            "output_dim": output dimensionality of model
            "num_gcn": number of graph layers
            "num_dense": number of dense layers
            "hidden_channels": hidden size for graph layers
            "dense_hidden": hidden size for dense layers
            "dropout_p": dropout probability
            "gnn_layer": type of graph layer to be used   
    config: dictionary, arguments related to the task
        "network": network target prediction
        "epochs": number of training epochs
        "lr_decay": lr decay
    save_dir: path to which the model should be saved
"""
def save_model(model, model_args, config, save_dir):
    torch.save({"model_args": model_args,
                "config": config,
                "model_state_dict": model.state_dict()},
                save_dir)
    
"""
Loads Model
params:
    model_class: model class name from which we can initialize
    save_dir: path to saved model
returns:
    model: saved model
    model_args: dictionary describing model (see above)
    config: dictionary describing model training (see above)
"""
def load_model(model_class, save_dir):
    checkpoint = torch.load(save_dir, weights_only=True)
    
    config = checkpoint["config"]

    model_args = checkpoint["model_args"]
    model = model_class(**model_args)

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, model_args, config