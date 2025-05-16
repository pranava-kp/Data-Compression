from model_utils import load_models, preprocess_data
import gradio as gr

# Load pre-trained models with their encoders
dt_bundle, rf_bundle = load_models()

# Prediction function
def predict(file_obj, model_choice):
    try:
        file_path = file_obj.name
        
        # Preprocess the uploaded file
        processed_data = preprocess_data(file_path)
        if processed_data is None:
            return "Error: Could not process file"

        # Select the model bundle
        bundle = dt_bundle if model_choice == "Decision Tree" else rf_bundle
        model = bundle['model']
        le_ext = bundle['le_ext']
        le_target = bundle['le_target']

        # Encode file extension if encoder is available
        ext = processed_data['file_extension'][0]
        if le_ext and ext in le_ext.classes_:
            processed_data['file_extension'] = le_ext.transform([ext])[0]
        else:
            processed_data['file_extension'] = 0  # Default value for unknown extensions

        # Make prediction
        prediction_encoded = model.predict(processed_data)
        prediction = le_target.inverse_transform(prediction_encoded)

        return f"Predicted best compression tool: {prediction[0]}"

    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check the file type and try again."


# === Gradio UI using Blocks ===
with gr.Blocks(theme=gr.themes.Base(), css="""
    #submit-btn {
        background-color: #ff6600 !important;
        color: white !important;
        font-weight: bold;
    }
""") as demo:
    gr.Markdown(
        """
        <div style='text-align: center;'>
            <h1 style='font-size: 36px; margin-bottom: 0;'>File Compression Predictor</h1>
            <p style='font-size: 18px; margin-top: 4px;'>Upload any file to predict the best compression tool to use (gzip, bzip2, etc.)</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload File")
            model_choice = gr.Radio(["Decision Tree", "Random Forest"], label="Select Model")
            with gr.Row():
                submit_btn = gr.Button("Submit", elem_id="submit-btn")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=1):
            output_box = gr.Textbox(label="Output")

    submit_btn.click(fn=predict, inputs=[file_input, model_choice], outputs=output_box)
    clear_btn.click(
        fn=lambda: (None, None, ""),
        inputs=[],
        outputs=[file_input, model_choice, output_box]
    )


demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)