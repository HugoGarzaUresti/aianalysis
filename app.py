from flask import Flask, json, request, jsonify, send_file
from flask.helpers import send_from_directory
from flask_cors import CORS, cross_origin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.stattools import acf
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np
import os
from werkzeug.utils import secure_filename
import csv
import pandas as pd

app = Flask(__name__, static_folder="frontend/build", static_url_path="")
cors = CORS(app)

data_type = {}
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(["csv"])

def check_valid_csv(filename, date_format="%Y-%m-%d"):
    try: pd.read_csv(filepath_or_buffer=filename)
    except Exception as e: return str(e)
    
    return None

def read_csv_wrapper(filename, date_format="%Y-%m-%d"):
    df = pd.read_csv(filepath_or_buffer=filename)
    
    for col_name, col_data in df.items():
        if pd.api.types.is_numeric_dtype(col_data):
            continue
        
        try:
            new_data = pd.to_datetime(col_data, format=date_format)
            new_data = pd.to_numeric(new_data) // (864 * 10 ** 11) # nanoseconds in a day
            df[col_name] = new_data
        except ValueError as e:
            app.logger.error(e)
            df = df.drop(columns=[col_name])

    return df

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
@cross_origin()
def serve():
    return send_from_directory(app.static_folder, "index.html")


def file_exists(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return os.path.exists(file_path)


@app.route("/api/v1/upload", methods=["POST"])
@cross_origin()
def upload_file():
    if "file" not in request.files:
        resp = jsonify({"message": "No file part in the request"})
        resp.status_code = 400
        return resp

    file = request.files["file"]

    errors = {}
    success = False

    full_path = None
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if file_exists(filename):
            errors[file.filename] = "File already exists"
        else:
            full_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(full_path)
            success = True
    else:
        errors[file.filename] = "File type is not allowed"

    err = check_valid_csv(full_path) if full_path is not None and success else None
    if err is not None:
        errors[file.filename] = err
        success = False
        os.remove(full_path)

    if success and errors:
        errors["message"] = "File successfully uploaded"
        resp = jsonify(errors)
        resp.status_code = 500
        return resp

    if success:
        resp = jsonify({"message": "File successfully uploaded"})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp


from flask import jsonify

@app.route("/api/v1/analyze", methods=["GET"])
@cross_origin()
def clean_data():
    operation = request.args.get("operation")
    FileName = request.args.get("filename")

    def clean_file(file_path, method):
        try:
            df = read_csv_wrapper(file_path)

            original_rows = len(df)

            if method == "clean":
                df.dropna(inplace=True)
                rows_changed = original_rows - len(df)
            elif method == "patch":
                original_df = df.copy()
                df.fillna(df.mean(), inplace=True)
                rows_changed = (original_df != df).any(axis=1).sum()
            elif method == "outliers":
                Q1 = df.quantile(0.25)
                Q3 = df.quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

                rows_changed = original_rows - len(df)
            elif method == "scale":
                # Scale the data to be between 0 and 1
                scaler = MinMaxScaler()
                df[df.columns] = scaler.fit_transform(df)
                rows_changed = original_rows
            elif method == "normalize":
                # Normalize the data to be between -1 and 1
                scaler = StandardScaler()
                df[df.columns] = scaler.fit_transform(df)
                rows_changed = original_rows
            elif method == "standardize":
                # Standardize the data to have a mean of 0 and standard deviation of 1
                scaler = StandardScaler()
                df[df.columns] = scaler.fit_transform(df)
                rows_changed = original_rows
            elif method == "pca":
                from sklearn.decomposition import PCA
                # Retain enough components to explain 95% of the variance
                pca = PCA(0.95)
                transformed_data = pca.fit_transform(df)
                # Convert the transformed data back to a dataframe
                df = pd.DataFrame(transformed_data, columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])])
                rows_changed = original_rows
            elif method == "bin":
                # Binning the 'x' column
                df['x_binned'] = pd.cut(df['x'], bins=10, labels=False)

                # Binning the 'y' column
                df['y_binned'] = pd.cut(df['y'], bins=10, labels=False)

                rows_changed = original_rows
            elif method == "discretize":
                # Binning the 'x' column into equal frequency bins
                df['x_binned'] = pd.qcut(df['x'], q=10, labels=False)

                # Binning the 'y' column into equal frequency bins
                df['y_binned'] = pd.qcut(df['y'], q=10, labels=False)

                rows_changed = original_rows
            elif method == "oneHotEncode":
                # Perform one-hot encoding on categorical data
                df = pd.get_dummies(df)
                rows_changed = original_rows
            elif method == "tsne":
                # Perform t-distributed Stochastic Neighbor Embedding
                # Note: You need to provide appropriate parameters for t-SNE
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2)
                transformed_data = tsne.fit_transform(df)
                df = pd.DataFrame(transformed_data, columns=['Dimension1', 'Dimension2'])
                rows_changed = original_rows
            elif method == "umap":
                # Perform Uniform Manifold Approximation and Projection
                # Note: You need to provide appropriate parameters for UMAP
                import umap
                umap_model = umap.UMAP(n_components=2)
                transformed_data = umap_model.fit_transform(df)
                df = pd.DataFrame(transformed_data, columns=['UMAP 1', 'UMAP 2'])
                rows_changed = len(df)
            elif method == "kmeans":
                # Perform K-Means Clustering
                # Note: You need to provide appropriate parameters for K-Means
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2)
                df['Cluster'] = kmeans.fit_predict(df)
                rows_changed = original_rows
            elif method == "dbscan":
                # Perform DBSCAN Clustering
                # Note: You need to provide appropriate parameters for DBSCAN
                from sklearn.cluster import DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                df['Cluster'] = dbscan.fit_predict(df)
                rows_changed = original_rows
            elif method == "hierarchical":
                # Perform Hierarchical Clustering
                # Note: You need to provide appropriate parameters for Hierarchical Clustering
                from sklearn.cluster import AgglomerativeClustering
                hierarchical = AgglomerativeClustering(n_clusters=2)
                df['Cluster'] = hierarchical.fit_predict(df)
                rows_changed = original_rows
            else:
                raise ValueError("Invalid cleaning method")

            df.to_csv(file_path, index=False)

            return rows_changed
        except Exception as e:
            return str(e)

    upload_folder = app.config["UPLOAD_FOLDER"]

    total_rows_changed = 0

    for filename in os.listdir(upload_folder):
        if filename == FileName:
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path) and file_path.endswith(".csv"):
                rows_changed = clean_file(file_path, operation)
                total_rows_changed = rows_changed

    response_message = f"Total rows changed: {total_rows_changed} ({operation} in {FileName})"
    print(response_message)

    resp = jsonify({"message": response_message})
    resp.status_code = 200
    return resp


@app.route("/api/v1/visualize", methods=["GET"])
@cross_origin()
def visualize_data():
    filename = request.args.get("filename", "")
    file = f"./static/uploads/{filename}"
    data = []

    try:
        df = read_csv_wrapper(file)
        for _, col in df.iterrows():
            data.append(col.to_dict())
        return jsonify(data)
    except:
        return jsonify({"error": "CSV file not found"})

    return jsonify(data)


@app.route("/api/v1/statistics", methods=["GET"])
@cross_origin()
def get_statistics():
    global data_type
    output = {}

    filename = request.args.get("filename", None)
    
    if filename is None:
        return jsonify({"error": "filename missing"}), 404
    
    if filename.endswith(".csv"):
        determine_data_type(filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            # Ensure the first row is used as header
            df = read_csv_wrapper(file_path)
            file_type = data_type.get(filename, "Unknown")
            stats_json = (
                df.describe(include="all").transpose().to_json(orient="index")
            )

            if file_type == "Linear Model":
                output = json.loads(stats_json)

            # other cases remain the same

            elif file_type == "Parabola":
                stats = json.loads(stats_json)
                # Dynamically handle column names
                X = df.iloc[:, :-1].values  # All columns except the last one
                y = df.iloc[:, -1].values  # Last column
                poly_reg = PolynomialFeatures(degree=2)
                X_poly = poly_reg.fit_transform(X)
                pol_reg = LinearRegression()
                pol_reg.fit(X_poly, y)
                a = pol_reg.coef_[2]
                b = pol_reg.coef_[1]
                c = pol_reg.intercept_

                vertex_x = -b / (2 * a)
                vertex_y = c - (b**2 / (4 * a))

                stats["Vertex"] = {"x": vertex_x, "y": vertex_y}
                output = stats

                return jsonify(output)
            elif file_type == "Unknown":
                # Compute basic statistics for x and y columns
                basic_stats = df.describe().to_dict()
                output = {
                    "Basic Statistics": {
                        "X": basic_stats[df.columns[0]],
                        "Y": basic_stats[df.columns[1]]
                    }
                }
            # other cases remain the same

        except FileNotFoundError:
            return jsonify({"error": f"{filename} not found"}), 404
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

    return jsonify(output)


def determine_data_type(filename):
    global data_type
    file_path = f"./static/uploads/{filename}"

    if not os.path.exists(file_path):
        data_type[filename] = "Unknown"
        return

    data = read_csv_wrapper(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Linear Regression Check
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        predictions = lin_reg.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        normalized_mse = mse / (y_test.max() - y_test.min())

        if normalized_mse < 0.1:
            data_type[filename] = "Linear Model"
            return
    except Exception as e:
        print(f"Error in Linear Regression check: {e}")
        
    # Parabola Check
    try:
        polynomial = make_pipeline(PolynomialFeatures(2), LinearRegression())
        polynomial.fit(X_train, y_train)
        poly_mse = mean_squared_error(y_test, polynomial.predict(X_test))

        if poly_mse < 0.5:
            data_type[filename] = "Parabola"
            return
    except Exception as e:
        print(f"Error in Parabola check: {e}")
    
    # Cluster Check
    try:
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        silhouette_score = np.mean(silhouette_samples(X, kmeans.labels_))
        if silhouette_score > 0.3:
            data_type[filename] = "Cluster"
            return
    except Exception as e:
        print(f"Error in Cluster check: {e}")

    # Time Series Check
    try:
        X.iloc[:, 0] = pd.to_datetime(
            X.iloc[:, 0]
        )  # Convert the first column to datetime
        autocorrelation = acf(y, nlags=40, fft=True)
        if autocorrelation[1] > 0.5:
            data_type[filename] = "Time Series"
            return
    except Exception as e:
        print(f"Error in Time Series check: {e}")
        
    data_type[filename] = "Unknown "
    return 

@app.route("/<path>", methods=["GET"])
@cross_origin()
def serve_index_html(path):
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/v1/files", methods=["GET"])
@cross_origin()
def fetch_files():
    files = []
    for file_name in [f for f in os.listdir(UPLOAD_FOLDER) if not f.startswith(".")]:
        file_path = os.path.join(UPLOAD_FOLDER, file_name)

        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            file_info = {"file_name": file_name, "file_size": file_size}
            files.append(file_info)
    resp = jsonify(files)
    return resp


@app.route("/api/v1/delete/<filename>", methods=["DELETE"])
@cross_origin()
def delete_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify(message="File deleted successfully"), 200
    else:
        return jsonify(error="File not found"), 404

#############################################################################################################
#############################################################################################################
#############################################################################################################

def export_data_to_pdf(data, filename, chart_type):
    pdf_pages = pdf_backend.PdfPages(filename)

    if chart_type == "scatter":
        plt.figure()
        x_values = [float(entry['x']) for entry in data]
        y_values = [float(entry['y']) for entry in data]
        plt.scatter(x_values, y_values)
        plt.title("Scatter Plot")
        pdf_pages.savefig()

    elif chart_type == "table":
        plt.figure()
        table_data = [list(data[0].keys())] + [[entry[key] for key in entry] for entry in data]
        table = plt.table(cellText=table_data, loc='center')
        pdf_pages.savefig()

    pdf_pages.close()


@app.route("/api/v1/export/pdf", methods=["GET"])
@cross_origin()
def export_to_pdf():
    filename = request.args.get("filename")
    chart_type = "scatter"
    pdf_filename = "exported_data.pdf"

    if os.path.exists(pdf_filename):
        os.remove(pdf_filename)

    data_response = visualize_data()  
    if data_response.status_code != 200:
        return data_response  

    try:
        response_data  = data_response.json 
    except Exception as e:
        return jsonify({"error": str(e)})

    export_data_to_pdf(response_data, pdf_filename, chart_type)
    return send_file(
        pdf_filename,
        mimetype='application/pdf',
        download_name=pdf_filename,
        as_attachment=True
    )

def export_statistics_to_csv(filename, stats):
    with open(filename, "w") as f:
        for section_name, section_data in stats.items():
            f.write(section_name + "\n")
            f.write("Stat,Value\n")
            for stat, value in section_data.items():
                f.write(f"{stat},{value}\n")

@app.route("/api/v1/export/csv", methods=["GET"])
@cross_origin()
def export_to_csv():
    filename = request.args.get("filename")
    stats_response = get_statistics()

    if stats_response.status_code != 200:
        return stats_response

    try:
        stats_data = stats_response.json
    except Exception as e:
        return jsonify({"error": str(e)})

    filenameDestination = "exported_statistics.csv"
    export_statistics_to_csv(filenameDestination, stats_data)

    return send_file(
        filenameDestination,
        mimetype='text/csv',
        download_name=filenameDestination,
        as_attachment=True
    )


@app.route("/api/v1/getColumns", methods=["GET"])
def get_file_column_names():
    filename = request.args.get("filename", None)
    if filename is None:
        return jsonify({"error": "missing file name"}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "file does not exist"}), 404
    
    try: 
        df = read_csv_wrapper(file_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
        
    return jsonify({"columns": df.columns.tolist()})

@app.route("/api/v1/scatterVisualiation", methods=["GET"])
def scatter_visualization():
    filename = request.args.get("filename", None)
    column_x = request.args.get("columnX", None)
    column_y = request.args.get("columnY", None)
    
    if filename is None or column_x is None or column_y is None:
        return jsonify({"error": "arguments missing"}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "file not found"}), 404
    
    try: df = read_csv_wrapper(file_path)
    except Exception as e: return jsonify({"error": str(e)}), 400
        
    if not (column_x in df.columns and column_y in df.columns):
        return jsonify({"error": "column names mismarch"}), 400
    
    data = []
    for _, row in df.iterrows():
        data.append({"x": row[column_x], "y": row[column_y]})
        
    return jsonify({"data": data})
    

def start():
    app.run()


if __name__ == "__main__":
    start()
