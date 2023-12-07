from ..global_dict import get_value
import torch

class FrameProcessor:
    def __init__(self) -> None:
        self.model = get_value('clip_model')
        self.processor = get_value('clip_img_preprocess')

    def text_probability_from_tensor_paths(self, serial_image_tensor_paths: str, text: str) -> str:
        image_tensor_paths = serial_image_tensor_paths.split(' ')

        avg_sum = 0.0
        for image_tensor_path in image_tensor_paths:
            image_tensor = torch.load(image_tensor_path).to(self.device)

            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)

            inputs['pixel_values'] = image_tensor    
            outputs = self.model(**inputs)

            logits_per_image = outputs.logits_per_image    
            probs = logits_per_image.squeeze()

            avg_sum += probs.item()

        return str(avg_sum / len(image_tensor_paths))