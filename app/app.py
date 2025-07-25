# app/app.py

import os
import uuid
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

from src.data_preparation import preprocess
from src.predict import predict_all

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "outputs", "predictions")
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Routes ---

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                # Read file
                ext = filename.rsplit('.', 1)[1].lower()
                if ext == "csv":
                    df = pd.read_csv(file_path)
                elif ext == "xlsx":
                    df = pd.read_excel(file_path)
                else:
                    return render_template("index.html", error="Supported file types are: CSV, XLSX")

                if "review_text" not in df.columns or "rating" not in df.columns:
                    return render_template("index.html", error="The file must contain 'review_text' and 'rating' columns.")

                # Run preprocessing and prediction pipeline
                df_clean = preprocess(df)
                print("📞 [APP] Calling predict_all function...")
                df_pred = predict_all(df_clean)

                # Sorts most suspicious (marked fake + negative), most positive (genuine + positive)
                top_suspicious = df_pred[df_pred['predicted_fake'] == 1].head(5)
                top_positive = df_pred[(df_pred['predicted_fake'] == 0) & (df_pred['predicted_sentiment'] == 'positive')].head(5)

                top_suspicious = top_suspicious[['review_text', 'predicted_fake', 'sentiment', 'corrected_rating']]
                top_positive = top_positive[['review_text', 'predicted_fake', 'predicted_sentiment', 'corrected_rating']]

                # Save output file
                output_id = str(uuid.uuid4())[:8]
                output_filename = f"predicted_reviews_{output_id}.csv"
                output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
                df_pred.to_csv(output_path, index=False)

                # Calculate stats for dashboard
                fake_pct = round((df_pred['predicted_fake'].sum() / len(df_pred)) * 100, 2)
                sentiment_counts = df_pred['predicted_sentiment'].value_counts().to_dict()
                avg_before = round(df_pred['rating'].mean(), 2)
                avg_after = round(df_pred['corrected_rating'].mean(), 2)

                return render_template("index.html",
                                       fake_pct=fake_pct,
                                       avg_before=avg_before,
                                       avg_after=avg_after,
                                       sentiment_counts=sentiment_counts,
                                       download_file=output_filename,
                                       top_suspicious=top_suspicious.to_dict(orient='records'),
                                       top_positive=top_positive.to_dict(orient='records')
                                    )

            except Exception as e:
                return render_template("index.html", error=f"❌ Internal error: {str(e)}")

        else:
            return render_template("index.html", error="Please upload a valid .csv or .xlsx file.")

    return render_template("index.html")


@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "❌ File not found.", 404


# --- App Runner ---

if __name__ == "__main__":
    app.run(debug=False)
