import torch

def save_model(model, model_args, config, save_dir):
    torch.save({"model_args": model_args,
                "config": config,
                "model_state_dict": model.state_dict()},
                save_dir)
    
def load_model(model_class, save_dir):
    checkpoint = torch.load(save_dir, weights_only=True)
    
    config = checkpoint["config"]

    model_args = checkpoint["model_args"]
    model = model_class(**model_args)

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, model_args, config