from geoclip import GeoCLIP

from .modeling_clip_surgery import CLIPSurgeryVisionTransformer
def get_geoclip(device: str, surgery: bool = False, return_all: bool = False):
    model = GeoCLIP()
    
    if surgery:
        # save the original vision model states
        config = model.image_encoder.CLIP.vision_model.config
        orig_state = model.image_encoder.CLIP.vision_model.state_dict()
        
        # create a new surgery vision model
        surgery_vision_model = CLIPSurgeryVisionTransformer(config)
    
        # load pretrained clip weights into the surgery encoder
        # this loads the weights allso for the submodules of the vision model
        # such as CLIPSurgeryEncoder, CLIPSurgeryEncoderLayer, CLIPSurgeryAttention, etc.
        surgery_vision_model.load_state_dict(orig_state, strict=False)
        # change the vision model to the surgery vision model
        model.image_encoder.CLIP.vision_model = surgery_vision_model
    model.to(device)
    if return_all:
        return model
    else:
        return model.location_encoder
    
    