# app.py
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

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Upload File"),
        gr.Radio(["Decision Tree", "Random Forest"], label="Select Model")
    ],
    outputs="text",
    title="File Compression Predictor",
    description="Upload any file to predict the best compression tool to use (gzip, bzip2, etc.)"
)
iface.launch(server_name="0.0.0.0", server_port=7860, debug=True)
# iface.launch(debug=True)