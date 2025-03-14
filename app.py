import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO


def run_yolo(image=None, video=None, model_name="yolov8n", image_size=640, conf_threshold=0.25):
    # Load YOLO model
    model = YOLO(model_name)
    
    if image is not None:
        # Process image
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None  # Convert BGR to RGB for display

    elif video is not None:
        # Process video
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        return None, output_path  # Return the annotated video path

    return None, None


# Define Gradio interface
def create_app():
    with gr.Blocks() as app:
        gr.Markdown("### YOLO Object Detection")
        
        with gr.Row():
            with gr.Column():
                input_type = gr.Radio(["Image", "Video"], label="Select Input Type", value="Image")
                image_input = gr.Image(label="Upload Image", visible=True, type="pil")
                video_input = gr.Video(label="Upload Video", visible=False)
                model_name = gr.Dropdown(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                                         label="Model", value="yolov8n")
                image_size = gr.Slider(320, 1280, step=32, value=640, label="Image Size")
                conf_threshold = gr.Slider(0.0, 1.0, step=0.05, value=0.25, label="Confidence Threshold")
                detect_button = gr.Button("Run Detection")
            
            with gr.Column():
                output_image = gr.Image(label="Detected Image", visible=True)
                output_video = gr.Video(label="Detected Video", visible=False)
        
        # Change visibility based on input type
        def update_inputs(input_type):
            if input_type == "Image":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
        
        input_type.change(fn=update_inputs, inputs=[input_type], 
                          outputs=[image_input, video_input, output_image, output_video])
        
        # Run inference
        def detect(input_type, image, video, model_name, image_size, conf_threshold):
            if input_type == "Image" and image is not None:
                return run_yolo(image=image, model_name=model_name, image_size=image_size, conf_threshold=conf_threshold)
            elif input_type == "Video" and video is not None:
                return run_yolo(video=video, model_name=model_name, image_size=image_size, conf_threshold=conf_threshold)
            return None, None
        
        detect_button.click(detect, 
                            inputs=[input_type, image_input, video_input, model_name, image_size, conf_threshold],
                            outputs=[output_image, output_video])
    
    return app


# Launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()
