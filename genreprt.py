import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from datetime import datetime
import google.generativeai as genai
import pdfkit
import numpy as np
from joblib import load


config = pdfkit.configuration(wkhtmltopdf=r"PATH_TO_wkhtmltopdf.exe")
model = load_model("MODEL_PATH")
scaler = load("SCALER_PATH")
def generate_pdf(input_data, api_response):
    pdf_content = f"""
    <html>
    <head><title>Water Quality Prediction Report</title></head>
    <body>
        <h1>Water Quality Prediction Report</h1>
        <h2>Input Values:</h2>
        <ul>
            <li>pH: {input_data[0]}</li>
            <li>Hardness: {input_data[1]}</li>
            <li>Solids: {input_data[2]}</li>
            <li>Chloramines: {input_data[3]}</li>
            <li>Sulfate: {input_data[4]}</li>
            <li>Conductivity: {input_data[5]}</li>
            <li>Organic Carbon: {input_data[6]}</li>
            <li>Trihalomethanes: {input_data[7]}</li>
            <li>Turbidity: {input_data[8]}</li>
        </ul>

        <h2>Water Potability Prediction: Non-Potable</h2>

        <h2>Generalized report:</h2>
        <p>{api_response}</p>

        <h3>Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
    </body>
    </html>
    """
    pdfkit.from_string(pdf_content, "PATH_TO_PDF_TO_BE_SAVED",configuration=config)
def send_to_gemini_api(input_data):
    prompt = """
    I have developed a machine learning model, and it predicts potability of water. By taking values:
    pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity, it tells whether the water is consumable by humans or not.
    I gave values such as:
    1. pH: {}
    2. Hardness: {}
    3. Solids: {}
    4. Chloramines: {}
    5. Sulfate: {}
    6. Conductivity: {}
    7. Organic Carbon: {}
    8. Trihalomethanes: {}
    9. Turbidity: {}
    As per the above values, the model predicted the water as non-potable/non-consumable. Please tell me which aspects from the above values may be responsible for water non-consumable,just guess based on the standard or average ranges of input the water should be.do not tell about model incabalities of prediction and low data,only tell why the water is unconsumable only considering the inputs 
    """.format(input_data[0], input_data[1], input_data[2], input_data[3], input_data[4], input_data[5], input_data[6],
               input_data[7], input_data[8])

    genai.configure(api_key="YOUR_API_KEY")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def predict_water_quality():
    try:
        input_data = [float(entry.get()) for entry in entries]
        new_data = np.array(input_data)
        new_data_scaled = scaler.transform(new_data.reshape(1, -1))
        prediction = model.predict(new_data_scaled)
        if prediction > 0.5:
            messagebox.showinfo("Result", "The water is potable.")
        else:
            messagebox.showwarning("Result", "The water is not potable.")

            api_response = send_to_gemini_api(input_data)
            generate_pdf(input_data, api_response)

            download_button.grid()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def download_pdf():
    import os
    os.startfile('PATH_TO_PDF_TO_BE_SAVED')


root = tk.Tk()
root.title("Water Quality Prediction")

labels = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic Carbon', 'Trihalomethanes',
          'Turbidity']
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)

predict_button = tk.Button(root, text="Predict", command=predict_water_quality)
predict_button.grid(row=9, column=0, columnspan=2)

download_button = tk.Button(root, text="Download PDF", command=download_pdf)
download_button.grid(row=10, column=0, columnspan=2)
download_button.grid_remove()


root.mainloop()