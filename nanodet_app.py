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
        """Custom visualization that shows only class names without percentages"""
        import numpy as np
        
        result_img = meta["raw_img"][0].copy()
        
        # Comprehensive debugging
        print(f"=== DETECTION DEBUG INFO ===")
        print(f"Detection type: {type(dets)}")
        print(f"Class names: {class_names}")
        print(f"Score threshold: {score_thres}")
        
        if hasattr(dets, '__len__'):
            print(f"Detection length: {len(dets)}")
            
        if isinstance(dets, (list, tuple)):
            for i, item in enumerate(dets):
                print(f"Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'no shape')}")
                if hasattr(item, 'shape') and len(item.shape) > 0:
                    print(f"  Content sample: {item[:min(3, len(item))] if len(item) > 0 else 'empty'}")
        
        try:
            detections_found = False
            
            # Try multiple approaches to extract detections
            detection_arrays = []
            
            # Approach 1: Direct array
            if hasattr(dets, 'shape') and len(dets.shape) >= 2:
                detection_arrays.append(dets)
                print("Found direct detection array")
            
            # Approach 2: List/tuple of arrays
            elif isinstance(dets, (list, tuple)):
                for i, item in enumerate(dets):
                    if hasattr(item, 'shape') and len(item.shape) >= 2:
                        detection_arrays.append(item)
                        print(f"Found detection array at index {i}")
                    elif hasattr(item, '__len__') and len(item) > 0:
                        # Convert to numpy if possible
                        try:
                            arr = np.array(item)
                            if len(arr.shape) >= 2:
                                detection_arrays.append(arr)
                                print(f"Converted item {i} to detection array")
                        except:
                            pass
            
            # Process each detection array
            for det_array in detection_arrays:
                # Convert tensor to numpy if needed
                if hasattr(det_array, 'cpu'):
                    det_array = det_array.cpu().numpy()
                elif hasattr(det_array, 'numpy'):
                    det_array = det_array.numpy()
                
                print(f"Processing array with shape: {det_array.shape}")
                
                if len(det_array.shape) >= 2 and det_array.shape[1] >= 5:
                    # Filter by score threshold
                    scores = det_array[:, 4]
                    valid_mask = scores >= score_thres
                    valid_detections = det_array[valid_mask]
                    
                    print(f"Valid detections after filtering: {len(valid_detections)}")
                    
                    for j, detection in enumerate(valid_detections):
                        detections_found = True
                        
                        # Extract coordinates and score
                        x1, y1, x2, y2 = detection[:4]
                        score = detection[4]
                        
                        # Handle different class ID formats
                        if len(detection) > 5:
                            class_id = int(detection[5])
                        else:
                            # If no class ID, try to infer or use default
                            class_id = 0
                        
                        print(f"Detection {j}: bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), score={score:.3f}, class_id={class_id}")
                        
                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Ensure coordinates are valid
                        h, w = result_img.shape[:2]
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(x1+1, min(x2, w-1))
                        y2 = max(y1+1, min(y2, h-1))
                        
                        # Draw bounding box
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Get class name
                        if 0 <= class_id < len(class_names):
                            class_name = class_names[class_id]
                        else:
                            class_name = f"Class_{class_id}"
                        
                        print(f"Drawing label: {class_name}")
                        
                        # Draw label
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        
                        # Get text size
                        (text_w, text_h), baseline = cv2.getTextSize(
                            class_name, font, font_scale, thickness
                        )
                        
                        # Draw background rectangle
                        cv2.rectangle(result_img, 
                                    (x1, y1 - text_h - baseline - 5),
                                    (x1 + text_w + 10, y1), 
                                    (0, 255, 0), -1)
                        
                        # Draw text
                        cv2.putText(result_img, class_name, 
                                  (x1 + 5, y1 - baseline - 2),
                                  font, font_scale, (0, 0, 0), thickness)
            
            if not detections_found:
                print("No valid detections found - drawing simple message")
                # Instead of fallback, draw a message on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "No damage detected"
                cv2.putText(result_img, text, (50, 50), font, 1, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()
            # Just return the original image if there's an error
            
        print("=== END DEBUG INFO ===")
        return result_img

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
