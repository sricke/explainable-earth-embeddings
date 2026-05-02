import torch

@torch.no_grad()
def _get_loc_queue_embeddings(loc_queue, model, device) -> torch.Tensor:
    queue = loc_queue.get().to(device)
    return model.location_model_predict(queue)


@torch.no_grad()
def _get_text_queue_embeddings(text_queue, model, device) -> torch.Tensor:
    queue = text_queue.get().to(device)
    return model.text_model_predict(queue)