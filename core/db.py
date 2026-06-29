"""
Database Module for the Disease Prediction System.

SQLite-backed storage for all predictions with helper functions
for CRUD operations and statistics.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_connection():
    """
    Get a SQLite database connection with row_factory for dict-like access.

    Returns:
        sqlite3.Connection
    """
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enables accessing columns by name
    return conn


def init_db():
    """
    Initialize the database: create the predictions table if it doesn't exist.
    Called on application startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            disease_type TEXT NOT NULL,
            patient_age INTEGER,
            patient_data TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            shap_chart_path TEXT,
            lime_chart_path TEXT,
            lime_text TEXT
        )
    """)

    # Create an index on timestamp for efficient date-range queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
        ON predictions (timestamp)
    """)

    # Create an index on disease_type for filtering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_disease
        ON predictions (disease_type)
    """)

    conn.commit()
    conn.close()
    print(f"  ✓ Database initialized at {config.DATABASE_PATH}")


def save_prediction(data):
    """
    Save a prediction record to the database.

    Args:
        data: dict with keys:
            - disease_type (str): e.g., 'diabetes'
            - patient_age (int): patient's age
            - patient_data (dict): all input values as JSON-serializable dict
            - prediction (str): e.g., 'Diabetic'
            - confidence (float): 0.0 to 1.0
            - risk_level (str): 'Low', 'Medium', or 'High'
            - shap_chart_path (str): filename of SHAP chart
            - lime_chart_path (str): filename of LIME chart
            - lime_text (list): list of explanation strings

    Returns:
        int: ID of the newly inserted row
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO predictions
                (disease_type, patient_age, patient_data, prediction,
                 confidence, risk_level, shap_chart_path, lime_chart_path, lime_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("disease_type"),
                data.get("patient_age"),
                json.dumps(data.get("patient_data", {})),
                data.get("prediction"),
                data.get("confidence"),
                data.get("risk_level"),
                data.get("shap_chart_path"),
                data.get("lime_chart_path"),
                json.dumps(data.get("lime_text", [])),
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        print(f"  ✓ Prediction saved (id={row_id})")
        return row_id

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_prediction_by_id(pred_id):
    """
    Retrieve a single prediction by its ID.

    Args:
        pred_id: Integer prediction ID

    Returns:
        dict or None: Prediction record as a dictionary
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return _row_to_dict(row)


def get_all_predictions(filters=None):
    """
    Retrieve all predictions with optional filters.

    Args:
        filters: dict with optional keys:
            - disease_type (str): filter by disease
            - prediction (str): filter by outcome
            - date_from (str): ISO date string, inclusive
            - date_to (str): ISO date string, inclusive
            - limit (int): max results
            - offset (int): for pagination

    Returns:
        list: List of prediction dictionaries, ordered by timestamp DESC
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if filters:
        if filters.get("disease_type"):
            query += " AND disease_type = ?"
            params.append(filters["disease_type"])

        if filters.get("prediction"):
            query += " AND prediction = ?"
            params.append(filters["prediction"])

        if filters.get("date_from"):
            query += " AND timestamp >= ?"
            params.append(filters["date_from"])

        if filters.get("date_to"):
            query += " AND timestamp <= ?"
            params.append(filters["date_to"] + " 23:59:59")

    query += " ORDER BY timestamp DESC"

    if filters and filters.get("limit"):
        query += " LIMIT ?"
        params.append(filters["limit"])
        if filters.get("offset"):
            query += " OFFSET ?"
            params.append(filters["offset"])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [_row_to_dict(row) for row in rows]


def get_stats():
    """
    Get aggregate statistics for the dashboard.

    Returns:
        dict: {
            'total_predictions': int,
            'disease_distribution': dict (disease_type → count),
            'outcome_distribution': dict (prediction → count),
            'avg_confidence': float,
            'risk_distribution': dict (risk_level → count),
            'recent_predictions': list (last 5),
            'predictions_by_date': list of {date, count} dicts
        }
    """
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    # Total predictions
    cursor.execute("SELECT COUNT(*) as count FROM predictions")
    stats["total_predictions"] = cursor.fetchone()["count"]

    # Disease distribution
    cursor.execute(
        "SELECT disease_type, COUNT(*) as count FROM predictions GROUP BY disease_type"
    )
    stats["disease_distribution"] = {
        row["disease_type"]: row["count"] for row in cursor.fetchall()
    }

    # Outcome distribution
    cursor.execute(
        "SELECT prediction, COUNT(*) as count FROM predictions GROUP BY prediction"
    )
    stats["outcome_distribution"] = {
        row["prediction"]: row["count"] for row in cursor.fetchall()
    }

    # Average confidence
    cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
    avg = cursor.fetchone()["avg_conf"]
    stats["avg_confidence"] = round(avg, 4) if avg else 0

    # Risk level distribution
    cursor.execute(
        "SELECT risk_level, COUNT(*) as count FROM predictions GROUP BY risk_level"
    )
    stats["risk_distribution"] = {
        row["risk_level"]: row["count"] for row in cursor.fetchall()
    }

    # Predictions per day (last 30 days)
    cursor.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT 30
    """)
    stats["predictions_by_date"] = [
        {"date": row["date"], "count": row["count"]} for row in cursor.fetchall()
    ]

    conn.close()
    return stats


def delete_prediction(pred_id):
    """
    Delete a prediction record by ID.

    Args:
        pred_id: Integer prediction ID

    Returns:
        bool: True if deleted, False if not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return deleted


def _row_to_dict(row):
    """
    Convert a sqlite3.Row to a plain dictionary with parsed JSON fields.

    Args:
        row: sqlite3.Row object

    Returns:
        dict: Prediction record
    """
    d = dict(row)

    # Parse JSON fields
    if d.get("patient_data"):
        try:
            d["patient_data"] = json.loads(d["patient_data"])
        except (json.JSONDecodeError, TypeError):
            d["patient_data"] = {}

    if d.get("lime_text"):
        try:
            d["lime_text"] = json.loads(d["lime_text"])
        except (json.JSONDecodeError, TypeError):
            d["lime_text"] = []

    return d
