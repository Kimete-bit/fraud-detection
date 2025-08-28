import gradio as gr
import joblib
import pandas as pd

PIPE_PATH = "artifacts/fraud_pipeline.joblib"
META_PATH = "artifacts/metadata.txt"

# load pipeline + version (me fallback)
pipe = joblib.load(PIPE_PATH)
version = "unknown"
try:
    with open(META_PATH) as f:
        for line in f:
            if line.startswith("model_version="):
                version = line.strip().split("=", 1)[1]
                break
except Exception:
    pass

def predict(amount=1000.0, oldbalanceOrg=5000.0, newbalanceOrig=4000.0, txn_type="CASH_OUT", threshold=0.5):
    df = pd.DataFrame([{
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "type": str(txn_type)
    }])
    proba = float(pipe.predict_proba(df)[0, 1])
    label = "FRAUD" if proba >= float(threshold) else "NOT FRAUD"
    return {
        "Fraud probability": round(proba, 6),
        "Decision (@{:.2f})".format(float(threshold)): label,
        "Model version": version
    }

with gr.Blocks() as demo:
    gr.Markdown("# Fraud Detection – PaySim Demo\nLightGBM pipeline (OneHot on type) trained on PaySim dataset.")

    amount = gr.Number(label="Amount", value=1000)
    oldb   = gr.Number(label="Old Balance", value=5000)
    newb   = gr.Number(label="New Balance", value=4000)
    ttype  = gr.Dropdown(
        choices=["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"],
        value="CASH_OUT", label="Type"
    )
    thr = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="Decision threshold")

    submit = gr.Button("Submit")
    out = gr.JSON(label="Prediction")

    submit.click(predict, inputs=[amount, oldb, newb, ttype, thr], outputs=out)

    gr.Examples(
        examples=[
            [9000, 9000, 0, "TRANSFER", 0.50],
            [6000, 6000, 0, "CASH_OUT", 0.50],
            [200, 1000, 1200, "CASH_IN", 0.50],
        ],
        inputs=[amount, oldb, newb, ttype, thr],
        label="Examples"
    )

if __name__ == "__main__":
    demo.launch()
