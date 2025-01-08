import torch
import clip


class prompt_order():
    def __init__(self) -> None:
        super().__init__()
        self.template_list = [
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of the {}.",
            "a dark photo of a {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a blurry photo of the {}.",
            "a blurry photo of a {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection."
        ]

    def prompt(self, class_name):
        input_ensemble_template = [template.format(class_name) for template in self.template_list]
        return input_ensemble_template
    

def make_text_embedding(model, device, text=None, type_list=None, class_adaption=False, template_adaption=False):
    text_generator = prompt_order()

    with torch.no_grad():
        if class_adaption:
            text_feature_arr = []
            for type in type_list:
                if template_adaption:
                    text_list = text_generator.prompt(type)
                    texts = clip.tokenize(text_list).to(device)
                else:
                    text_list = [type for _ in range(1)]
                    texts = clip.tokenize(text_list).to(device)
                text_features = model.encode_text(texts).float()
                avg_text_feature = torch.mean(text_features, dim = 0, keepdim= True) 
                text_feature_arr.append(avg_text_feature)
            text_embedding = torch.stack(text_feature_arr).squeeze(1)

        else:
            if template_adaption:
                text_list = text_generator.prompt(text)
                texts = clip.tokenize(text_list).to(device)
            else:
                text_list = [text for _ in range(1)]
                texts = clip.tokenize(text_list).to(device)
            text_features = model.encode_text(texts).float()
            text_embedding = torch.mean(text_features, dim = 0, keepdim= True) 
    return text_embedding