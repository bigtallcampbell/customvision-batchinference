"""
Custom Vision batch inference example
"""
import datetime
import argparse
import os
import cv2
import json
import numpy as np
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

from app_config import AppConfig
APP_CONFIG:AppConfig = AppConfig()

def prerequisites():
    os.makedirs("./output", exist_ok=True)

def main():
    """
    Main entry point for the app
    """

    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": APP_CONFIG.CUSTOM_VISION_KEY})
    predictor = CustomVisionPredictionClient(APP_CONFIG.CUSTOM_VISION_ENDPOINT, prediction_credentials)
    

    for (root, dirs, files) in os.walk(APP_CONFIG.SOURCE_FOLDER):
        for file in files:
            if '.jpg' in file:
                print("Analyzing:", os.path.join(root, file))

                with open(os.path.join(root, file), mode="rb") as img_data:
                    results = predictor.detect_image(project_id=APP_CONFIG.CUSTOM_VISION_PROJECT_ID, published_name=APP_CONFIG.CUSTOM_VISION_PUBLISHED_ITERATION_NAME, image_data=img_data)

                    raw_img = cv2.imread(os.path.join(root, file))
                    raw_img_height = raw_img.shape[0]
                    raw_img_width = raw_img.shape[1]
                    
                    for prediction in results.predictions:
                        if prediction.probability < APP_CONFIG.PROBABILIY_THRESHOLD:
                            continue

                        #Bounding box is returned as a percentage of the image size.  Convert that back to the pixel number
                        boundingBox_coordinate_left = round(prediction.bounding_box.left * raw_img_width)
                        boundingBox_coordinate_width = round(prediction.bounding_box.width * raw_img_width)
                        boundingBox_coordinate_top = round(prediction.bounding_box.top * raw_img_height)
                        boundingBox_coordinate_height = round(prediction.bounding_box.height * raw_img_height)

                        hitbox_start_point = (boundingBox_coordinate_left, boundingBox_coordinate_top)
                        hitbox_end_point = (hitbox_start_point[0] + boundingBox_coordinate_width, hitbox_start_point[1] + boundingBox_coordinate_height)
                        color = (0, 0, 255)

                        hitbox_thickness = 15
                        
                        cv2.rectangle(raw_img, hitbox_start_point, hitbox_end_point, color, hitbox_thickness)

                        
                        tag_label = "{0:10}({1:.2%})".format(prediction.tag_name, prediction.probability)
                        

                        raw_img, _ = draw_text(raw_img, tag_label, font_scale=1.5, pos=hitbox_start_point, text_color_bg=(0, 0, 255), font_thickness=2, bg_offset_h=hitbox_thickness, bg_offset_w=hitbox_thickness)


                
                    filename = f"./output/{os.path.basename(file)}"
                    cv2.imwrite(filename, raw_img)

                        
def draw_text(img, text,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                pos=(0, 0),
                font_scale=3,
                font_thickness=1,
                text_color=(255, 255, 255),
                text_color_bg=(0, 0, 0),
                bg_offset_h=0,
                bg_offset_w=0
                ):
    """
    Helper function to write text to the image.
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    pos = (x, (y - text_h))
    x, y = pos

    bg_pos = (x - bg_offset_w, y - bg_offset_h)

    cv2.rectangle(img, bg_pos, (x + text_w, y + (text_h * 2)),
                    text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + 5), font,
                font_scale, text_color, font_thickness)
    text_end_pos = (x + text_w, y + (text_h * 2))

    return img, text_end_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter_file',
                        required=True,
                        dest='parameter_file',
                        help='The paramater input file to load the credentials and values for',
                        type=str
                        )





    args = parser.parse_args()

    print("------------------------------------------")
    print("Custom Vision - Batch Inference")
    now = datetime.datetime.now()
    print("Start time: ", now.strftime('%Y-%m-%d %H:%M:%S'))
    print("CONFIG VALUES: ")

    TEMPLATE = "{0:45}{1}"  # column widths: 15, n

    for key, value in sorted(vars(args).items()):
        print(TEMPLATE.format(key, value))

    print("Importing Parameter File...")

    if os.path.isfile(args.parameter_file) is False:
        error_string = f"Parameter file '{args.parameter_file}' doesn't exist.  Please check your path"
        raise Exception(error_string)

    with open(args.parameter_file) as f:
        parameter_file_data = json.load(f)

    if "CUSTOM_VISION_ENDPOINT" in parameter_file_data:
        APP_CONFIG.CUSTOM_VISION_ENDPOINT = parameter_file_data["CUSTOM_VISION_ENDPOINT"]
    else:
        error_string = f"Parameter file '{args.parameter_file}' missing 'CUSTOM_VISION_ENDPOINT' parameter.  Please check"
        raise Exception(error_string)

    if "CUSTOM_VISION_PROJECT_ID" in parameter_file_data:
        APP_CONFIG.CUSTOM_VISION_PROJECT_ID = parameter_file_data["CUSTOM_VISION_PROJECT_ID"]
    else:
        error_string = f"Parameter file '{args.parameter_file}' missing 'CUSTOM_VISION_PROJECT_ID' parameter.  Please check"
        raise Exception(error_string)

    if "CUSTOM_VISION_KEY" in parameter_file_data:
        APP_CONFIG.CUSTOM_VISION_KEY = parameter_file_data["CUSTOM_VISION_KEY"]
    else:
        error_string = f"Parameter file '{args.parameter_file}' missing 'CUSTOM_VISION_KEY' parameter.  Please check"
        raise Exception(error_string)

    
    if "CUSTOM_VISION_PUBLISHED_ITERATION_NAME" in parameter_file_data:
        APP_CONFIG.CUSTOM_VISION_PUBLISHED_ITERATION_NAME = parameter_file_data["CUSTOM_VISION_PUBLISHED_ITERATION_NAME"]
    else:
        error_string = f"Parameter file '{args.parameter_file}' missing 'CUSTOM_VISION_PUBLISHED_ITERATION_NAME' parameter.  Please check"
        raise Exception(error_string)

    
    if "PROBABILIY_THRESHOLD" in parameter_file_data:
        APP_CONFIG.PROBABILIY_THRESHOLD = parameter_file_data["PROBABILIY_THRESHOLD"]
    else:
        error_string = f"Parameter file '{args.parameter_file}' missing 'PROBABILIY_THRESHOLD' parameter.  Please check"
        raise Exception(error_string)


    if "SOURCE_FOLDER" in parameter_file_data:
        APP_CONFIG.SOURCE_FOLDER = parameter_file_data["SOURCE_FOLDER"]
        if os.path.isdir(parameter_file_data["SOURCE_FOLDER"]) is False:
            error_string = f"Input directory '{APP_CONFIG.SOURCE_FOLDER}' doesn't exist"
            raise Exception(error_string)
    else:
        error_string = f"Parameter file '{args.parameter_file}' missing 'SOURCE_FOLDER' parameter.  Please check"
        raise Exception(error_string)

    print("Parameter file imported.")
   
    print("PARAMETER FILE VALUES: ")
    PARAM_VALUE_TEMPLATE = "{0:45}{1}"  # column widths: 15, n
   
    print(PARAM_VALUE_TEMPLATE.format("CUSTOM_VISION_ENDPOINT", APP_CONFIG.CUSTOM_VISION_ENDPOINT))
    print(PARAM_VALUE_TEMPLATE.format("SOURCE_FOLDER", APP_CONFIG.SOURCE_FOLDER))
    print(PARAM_VALUE_TEMPLATE.format("PROBABILIY_THRESHOLD", APP_CONFIG.PROBABILIY_THRESHOLD))
    print(PARAM_VALUE_TEMPLATE.format("CUSTOM_VISION_PUBLISHED_ITERATION_NAME", APP_CONFIG.CUSTOM_VISION_PUBLISHED_ITERATION_NAME))
    print(PARAM_VALUE_TEMPLATE.format("CUSTOM_VISION_ENDPOINT", APP_CONFIG.CUSTOM_VISION_ENDPOINT))


    param_length=((APP_CONFIG.CUSTOM_VISION_KEY).join(APP_CONFIG.CUSTOM_VISION_KEY)).count(APP_CONFIG.CUSTOM_VISION_KEY) + 1
    obscured_vision_key=APP_CONFIG.CUSTOM_VISION_KEY[:5]
    obscured_vision_key+= "*" * (param_length - 5)
    print(PARAM_VALUE_TEMPLATE.format("CUSTOM_VISION_KEY", obscured_vision_key))


    prerequisites()
    main()

