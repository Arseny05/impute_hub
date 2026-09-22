import os
import pandas as pd
from flask import Flask, jsonify, request, Response
from DataManager import DatasetManager, StorageService, MetricsManager, MetricsRow
from dotenv import load_dotenv
import pickle

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

STORAGE_PATH = os.getenv("STORAGE_PATH", os.path.join(ROOT_DIR, "shared_data"))
DB_PATH = os.getenv("DB_PATH", os.path.join(STORAGE_PATH, "database.db"))
PORT = int(os.getenv("STORAGE_PORT", 5001))
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1")

storage_service = StorageService(db_path=DB_PATH, storage_path=STORAGE_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'OK', 'service':"storage_service"}), 200

@app.route('/api/storage/upload', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files or 'config' not in request.files:
        return jsonify({'error': 'file and config are necessary!'}), 400
    data_file = request.files['file']
    config_file = request.files['config']
    if not data_file.filename or not config_file.filename:
        return jsonify({"error": "Names of passed files should not be empty!"}), 400
    try:
        dataset_id = storage_service.register_data(config=config_file, data=data_file)
        return jsonify({"status":'success', "dataset_id":dataset_id, "message":"Dataset and configuration are loaded"}), 201
    except ValueError as e:
        return jsonify({"error": f"Error in schema validation: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error occured while data were loading: {str(e)}"}), 500

@app.route('/api/storage/metrics/<int:dataset_id>', methods=["GET"])
def get_metrics(dataset_id:int):
    try:
        with MetricsManager(DB_PATH) as m:
            record = m[dataset_id]
            return jsonify(record), 200
    except KeyError:
        return jsonify({'error': f"Metrics for dataset with ID {dataset_id} are not found!"}), 404
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/storage/metrics", methods=["POST"])
def save_metrics():
    data = request.get_json(silent=True)
    if not data or "dataset_id" not in data:
        return jsonify({'error': "dataset_id absents!"}), 400
    try:
        with DatasetManager(DB_PATH) as dm:
            _ = dm[data["dataset_id"]]

            row = MetricsRow(
                dataset_id=int(data["dataset_id"]),
                accuracy_weighted=data.get("accuracy_weighted"),
                precision_weighted=data.get("precision_weighted"),
                recall_weighted=data.get("recall_weighted"),
                f1_score_weighted=data.get("f1_score_weighted"),
                accuracy_macro=data.get("accuracy_macro"),
                precision_macro=data.get("precision_macro"),
                recall_macro=data.get("recall_macro"),
                f1_score_macro=data.get("f1_score_macro"),
                mse=data.get("mse"),
                mae=data.get("mae")
            )

            with MetricsManager(DB_PATH) as mm:
                mm.add(row)
            return jsonify({"status":"success", "dataset_id":row.dataset_id}), 201
    except KeyError:
        return jsonify({"error": f"Dataset with ID {data['dataset_id']} does not exist"}), 404
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/storage/dataset", methods=['GET'])
def get_dataset():
    dataset_id = request.args.get('dataset_id', type=int)
    name = request.args.get('name', type=str)
    status = request.args.get('status', default='o', type=str)
    fraction = request.args.get('fraction', default=0.0, type=float)
    as_file = request.args.get('as_file', default=True)
    as_file = str(as_file).lower() in ('true', '1')
    try:
        with DatasetManager(db_path=DB_PATH) as dm:
            if dataset_id is not None:
                data = dm[dataset_id]
            elif name is not None:
                data = dm[(name, status, fraction)]
            else:
                return jsonify({'error':'Missing necessary argument (dataset id or dataset name)'}), 400
        if as_file:
            file_path = data.get('file_path')
            if not file_path or not os.path.exists(file_path):
                return jsonify({'error':'No appropriate file in storage'}), 404
            df = pd.read_csv(file_path)
            res = {
                'metadata':data,
                'data': df
            }
            pickled_data = pickle.dumps(res)
            return Response(pickled_data, mimetype='application/octet-stream')
        else:
            return jsonify({'status':'success', 'data':data}), 200
    except KeyError as e:
        return jsonify(({'error': str(e)})), 404
    except Exception as e:
        return jsonify({'error':f'Internal server error {str(e)}'}), 500

@app.route('/api/storage/dataset', methods=['DELETE'])
def delete_dataset():
    dataset_id = request.args.get('dataset_id', type=int)
    name = request.args.get('name', type=str)
    status = request.args.get('status', default='o', type=str)
    fraction = request.args.get('fraction', default=0.0, type=float)
    try:
        with DatasetManager(db_path=DB_PATH) as dm:
            if dataset_id is not None:
                del dm[dataset_id]
                target = f'ID {dataset_id}'
            elif name is not None:
                del dm.delete_by_data(name, status, fraction)
                target = f'name={name}, table_class={status}, missing_rate={fraction}'
            else:
                return jsonify({'error':'Missing necessary argument (dataset id or dataset name)'}), 400
        return jsonify({'status':'success','data': f'Dataset with {target} has been deleted with its metrics'}), 200
    except KeyError as e:
        return jsonify(({'error': str(e)})), 404
    except Exception as e:
        return jsonify({'error':f'Internal server error {str(e)}'}), 500



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)





