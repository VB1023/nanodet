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
        # Custom visualization without percentage scores
        result_img = meta["raw_img"][0].copy()
        
        # Handle different detection result formats
        if isinstance(dets, list) and len(dets) > 0:
            # If dets is a list of detections per class
            for class_id, class_dets in enumerate(dets):
                if hasattr(class_dets, 'shape') and len(class_dets.shape) > 0:
                    # If it's a numpy array or tensor
                    if len(class_dets) > 0:
                        for detection in class_dets:
                            if len(detection) >= 5:
                                x1, y1, x2, y2, score = detection[:5]
                                if score >= score_thres:
                                    # Draw bounding box
                                    cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                    
                                    # Draw class name only (without percentage)
                                    if class_id < len(class_names):
                                        class_name = class_names[class_id]
                                        # Position label above the bounding box
                                        label_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 20
                                        cv2.putText(result_img, class_name, (int(x1), label_y), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif isinstance(class_dets, (list, tuple)) and len(class_dets) > 0:
                    # If it's a list/tuple of detections
                    for detection in class_dets:
                        if len(detection) >= 5:
                            x1, y1, x2, y2, score = detection[:5]
                            if score >= score_thres:
                                # Draw bounding box
                                cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                
                                # Draw class name only (without percentage)
                                if class_id < len(class_names):
                                    class_name = class_names[class_id]
                                    # Position label above the bounding box
                                    label_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 20
                                    cv2.putText(result_img, class_name, (int(x1), label_y), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
