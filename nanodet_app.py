import os
import time
import cv2
import torch
import streamlit as st
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

class Predictor:
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(torch.device("cpu")).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB immediately
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres):
        result_img = meta["raw_img"][0].copy()
        
        try:
            # Handle different detection result formats
            if isinstance(dets, (list, tuple)):
                # If dets is a list/tuple, process each detection
                for det_group in dets:
                    if hasattr(det_group, 'shape') and len(det_group.shape) > 0:
                        self._draw_detections(result_img, det_group, class_names, score_thres)
                    elif isinstance(det_group, (list, tuple)) and len(det_group) > 0:
                        for det in det_group:
                            if hasattr(det, 'shape') and len(det.shape) > 0:
                                self._draw_detections(result_img, det, class_names, score_thres)
            else:
                # If dets is a single array/tensor
                if hasattr(dets, 'shape') and len(dets.shape) > 0:
                    self._draw_detections(result_img, dets, class_names, score_thres)
                    
        except Exception as e:
            print(f"Custom visualization error: {e}")
            print(f"Detection format: {type(dets)}")
            if hasattr(dets, 'shape'):
                print(f"Detection shape: {dets.shape}")
            # Fallback to original method if custom fails
            try:
                result_img = self.model.head.show_result(
                    meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
                )
            except:
                pass
        
        return result_img
    
    def _draw_detections(self, img, detections, class_names, score_thres):
        """Helper method to draw detections on image without percentages"""
        import numpy as np
        
        # Convert to numpy if it's a tensor
        if hasattr(detections, 'cpu'):
            detections = detections.cpu().numpy()
        elif hasattr(detections, 'numpy'):
            detections = detections.numpy()
        
        if len(detections) == 0:
            return
            
        # Ensure detections is 2D
        if len(detections.shape) == 1:
            detections = detections.reshape(1, -1)
            
        if detections.shape[1] < 5:  # Need at least bbox + score
            return
            
        # Extract components
        bboxes = detections[:, :4]
        scores = detections[:, 4]
        
        # Handle labels - might be in column 5 or derived from other info
        if detections.shape[1] > 5:
            labels = detections[:, 5].astype(int)
        else:
            # If no label column, assume class 0 for now
            labels = np.zeros(len(detections), dtype=int)
        
        # Filter by score threshold
        valid_mask = scores >= score_thres
        if not np.any(valid_mask):
            return
            
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]
        
        # Draw each detection
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text (without percentage)
            if 0 <= label < len(class_names):
                label_text = class_names[label]
            else:
                label_text = f"Object"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img, label_text, (x1, y1 - baseline - 2), 
                       font, font_scale, (0, 0, 0), thickness)

def get_image_list(path):
    image_names = []
    if os.path.isdir(path):
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in [".jpg", ".jpeg", ".webp", ".bmp", ".png"]:
                    image_names.append(apath)
    else:
        image_names.append(path)
    return image_names

def run_inference_for_image(config_path, model_path, image_path, save_result=False, save_dir='./inference_results'):
    load_config(cfg, config_path)
    logger = Logger(local_rank=0, use_tensorboard=False)
    predictor = Predictor(cfg, model_path, logger, device="cpu")
    image_names = get_image_list(image_path)
    image_names.sort()

    current_time = time.localtime()
    if save_result:
        save_folder = os.path.join(save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank=0, path=save_folder)

    result_images = []
    for image_name in image_names:
        meta, res = predictor.inference(image_name)
        result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)  # Ensures RGB format
        
        if save_result:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))  # Convert back to BGR before saving
        
        result_images.append(result_image)

    return result_images

def main():
    st.title("Car Damage Assessment")
    config_path = 'config/nanodet-plus-m_416-yolo.yml'
    model_path = 'workspace/nanodet-plus-m_416/model_best/model_best.ckpt'
    save_dir = './inference_results'

    image_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png", "bmp", "webp"])
    camera_image = st.camera_input("Take a picture")

    image_path = None
    if image_file is not None:
        image_path = "./temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file.read())
    elif camera_image is not None:
        image_path = "./temp_camera_image.jpg"
        with open(image_path, "wb") as f:
            f.write(camera_image.getbuffer())

    save_result = st.checkbox("Save Inference Results", value=False)

    if image_path:
        with st.spinner("Running inference..."):
            result_images = run_inference_for_image(config_path, model_path, image_path, save_result, save_dir)
        if result_images:
            st.image(result_images[0], caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
