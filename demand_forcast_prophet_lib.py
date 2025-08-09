from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet import Prophet
from datetime import datetime
import os
import joblib
import uvicorn
import pymongo
from pymongo import MongoClient
import numpy as np
from typing import Optional, List, Dict
import logging

# Configuration (read from environment for deployment)
MONGO_URI = os.getenv("MONGO_URI", "")
DATABASE_NAME = os.getenv("DATABASE_NAME", "taniaWaters")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "orderData")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")
MODEL_FORMAT = 'joblib'

# CORS configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")  # comma-separated list or *
ALLOWED_ORIGINS_LIST = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]

# Create cache directory
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales Forecasting API - Clean Approach", version="3.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MongoDB client
mongo_client = None
db = None
collection = None

def initialize_mongodb():
    """Initialize MongoDB connection"""
    global mongo_client, db, collection
    try:
        if not MONGO_URI:
            raise ValueError("MONGO_URI is not set. Please configure the environment variable.")
        mongo_client = MongoClient(MONGO_URI)
        mongo_client.admin.command('ping')
        db = mongo_client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Create indexes for better performance
        collection.create_index([("soldToPartyName", 1)])
        collection.create_index([("itemDescription", 1)])
        collection.create_index([("billingDate", 1)])
        collection.create_index([("soldToPartyName", 1), ("itemDescription", 1)])
        collection.create_index([("soldToPartyName", 1), ("billingDate", 1)])
        
        logger.info("Successfully connected to MongoDB and created indexes")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        return False

def get_model_cache_key(customer: str, product: str = None, skip_fridays: bool = False) -> str:
    """Generate unique cache key for each model (include skip_fridays flag)"""
    key = f"customer_{customer.replace(' ', '_').replace('/', '_')}"
    if product:
        key += f"_product_{product.replace(' ', '_').replace('×', 'x').replace('/', '_')}"
    if skip_fridays:
        key += "_nofriday"
    return os.path.join(MODEL_CACHE_DIR, f"{key}.joblib")

def load_cached_model(cache_key: str):
    """Load cached model if exists and recent"""
    if os.path.exists(cache_key):
        try:
            model_info = joblib.load(cache_key)
            if (datetime.now() - model_info['trained_at']).days < 7:
                logger.info(f"Using cached model: {cache_key}")
                return model_info['model']
            else:
                logger.info(f"Cached model too old, will retrain")
        except Exception as e:
            logger.error(f"Failed to load cached model: {str(e)}")
    return None

def save_model_to_cache(model, cache_key: str):
    """Save model with metadata to cache"""
    model_info = {
        'model': model,
        'trained_at': datetime.now(),
        'model_type': 'Prophet',
        'format': MODEL_FORMAT
    }
    try:
        joblib.dump(model_info, cache_key)
        logger.info(f"Model saved to cache: {cache_key}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")

def get_customer_product_data(customer: str, product: str, skip_fridays: bool = False) -> pd.DataFrame:
    """Get data for specific customer + product combination"""
    try:
        query = {
            "soldToPartyName": customer,
            "itemDescription": product
        }
        
        cursor = collection.find(query, {
            "billingDate": 1,
            "createdOn": 1,
            "billedQuantity": 1,
            "_id": 0
        })
        
        data = list(cursor)
        if not data:
            raise ValueError(f"No data found for customer '{customer}' and product '{product}'")
        
        df = pd.DataFrame(data)
        
        # Use billingDate as primary, fallback to createdOn
        if 'billingDate' in df.columns and df['billingDate'].notna().any():
            df['ds'] = pd.to_datetime(df['billingDate'], format='%d/%m/%Y', errors='coerce')
        elif 'createdOn' in df.columns:
            df['ds'] = pd.to_datetime(df['createdOn'], format='%d/%m/%Y', errors='coerce')
        else:
            raise ValueError("No valid date field found")
        
        df['y'] = pd.to_numeric(df['billedQuantity'], errors='coerce')
        
        # Clean data
        df = df.dropna(subset=['ds', 'y'])
        df = df[df['y'] > 0]

        if skip_fridays:
            # Drop Saudi weekend days (Friday=4, Saturday=5)
            df = df[~df['ds'].dt.weekday.isin({4, 5})]
            logger.info(f"Filtered out Friday+Saturday (Saudi weekend), {len(df)} data points remain for {customer} - {product}")
        
        logger.info(f"Loaded {len(df)} data points for {customer} - {product}")
        return df[['ds', 'y']].sort_values('ds')
        
    except Exception as e:
        raise RuntimeError(f"Failed to get customer-product data: {str(e)}")

def get_customer_aggregated_data(customer: str, skip_fridays: bool = False) -> pd.DataFrame:
    """Get aggregated data for all products of a customer (sum quantities by date)"""
    try:
        query = {"soldToPartyName": customer}
        
        # Aggregation pipeline to sum quantities by date
        # Handle both string and datetime formats
        pipeline = [
            {"$match": query},
            {
                "$addFields": {
                    "date_field": {
                        "$cond": {
                            "if": {"$ne": ["$billingDate", None]},
                            "then": {
                                "$cond": {
                                    "if": {"$eq": [{"$type": "$billingDate"}, "date"]},
                                    "then": "$billingDate",
                                    "else": {
                                        "$dateFromString": {
                                            "dateString": "$billingDate",
                                            "format": "%d/%m/%Y"
                                        }
                                    }
                                }
                            },
                            "else": {
                                "$cond": {
                                    "if": {"$eq": [{"$type": "$createdOn"}, "date"]},
                                    "then": "$createdOn",
                                    "else": {
                                        "$dateFromString": {
                                            "dateString": "$createdOn",
                                            "format": "%d/%m/%Y"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$date_field"
                        }
                    },
                    "total_quantity": {"$sum": {"$toDouble": "$billedQuantity"}}
                }
            },
            {"$match": {"total_quantity": {"$gt": 0}}},
            {
                "$addFields": {
                    "date_obj": {
                        "$dateFromString": {
                            "dateString": "$_id",
                            "format": "%Y-%m-%d"
                        }
                    }
                }
            },
            {"$sort": {"date_obj": 1}}
        ]
        
        aggregated_data = list(collection.aggregate(pipeline))
        
        if not aggregated_data:
            raise ValueError(f"No data found for customer '{customer}'")
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {"ds": pd.to_datetime(item["_id"]), "y": item["total_quantity"]} 
            for item in aggregated_data
        ])
        
        if skip_fridays:
            df = df[~df['ds'].dt.weekday.isin({4, 5})]
            logger.info(f"Filtered out Friday+Saturday (Saudi weekend), {len(df)} aggregated data points remain for customer {customer}")
        
        logger.info(f"Loaded {len(df)} aggregated data points for customer {customer}")
        return df.sort_values('ds')
        
    except Exception as e:
        raise RuntimeError(f"Failed to get customer aggregated data: {str(e)}")

def train_customer_product_model(customer: str, product: str, force_retrain: bool = False, skip_fridays: bool = False):
    """Train model for specific customer + product"""
    cache_key = get_model_cache_key(customer, product, skip_fridays)
    
    if not force_retrain:
        cached_model = load_cached_model(cache_key)
        if cached_model:
            return cached_model
    
    # Get data and train model
    df = get_customer_product_data(customer, product, skip_fridays)
    
    if len(df) < 2:
        raise ValueError(f"Insufficient data points for training (found {len(df)}, need at least 2)")
    
    logger.info(f"Training model for {customer} - {product} with {len(df)} data points (skip_fridays={skip_fridays})")
    
    model = Prophet(
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=False
    )
    model.fit(df)
    
    save_model_to_cache(model, cache_key)
    return model

def train_customer_total_model(customer: str, force_retrain: bool = False, skip_fridays: bool = False):
    """Train model for customer total (aggregated across all products)"""
    cache_key = get_model_cache_key(customer, None, skip_fridays)  # include skip_fridays flag
    
    if not force_retrain:
        cached_model = load_cached_model(cache_key)
        if cached_model:
            return cached_model
    
    # Get aggregated data and train model
    df = get_customer_aggregated_data(customer, skip_fridays)
    
    if len(df) < 2:
        raise ValueError(f"Insufficient aggregated data points for training (found {len(df)}, need at least 2)")
    
    logger.info(f"Training aggregated model for {customer} with {len(df)} data points (skip_fridays={skip_fridays})")
    
    model = Prophet(
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=False
    )
    model.fit(df)
    
    save_model_to_cache(model, cache_key)
    return model

def _get_future_non_weekend_forecast(model: Prophet, horizon: int):
    """
    Helper: create forecasts for future days excluding Saudi weekend (Friday, Saturday).
    Ensures we return exactly `horizon` future rows that are not on weekend days (if possible).
    """
    # Start with a generous number of periods; expand if necessary
    periods = max(horizon * 2, horizon + 30)
    max_periods = max(horizon * 10, 365)
    tries = 0
    weekend_days = {4, 5}  # Friday=4, Saturday=5

    while True:
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        # get only future rows (ds > last training ds)
        last_train = model.history['ds'].max()
        future_pred = forecast[forecast['ds'] > last_train].copy()
        # filter out weekend days (Friday & Saturday)
        future_non_weekend = future_pred[~future_pred['ds'].dt.weekday.isin(weekend_days)]
        if len(future_non_weekend) >= horizon or periods >= max_periods:
            return future_non_weekend.head(horizon)
        # increase periods and retry
        periods = int(periods * 1.5) + 10
        tries += 1
        if tries > 10:
            # fallback: return what we have
            return future_non_weekend.head(horizon)

@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection on startup"""
    success = initialize_mongodb()
    if not success:
        raise Exception("MongoDB connection failed")

@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on shutdown"""
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

@app.get("/")
async def root():
    """Health check endpoint"""
    try:
        mongo_client.admin.command('ping')
        mongo_status = "connected"
    except:
        mongo_status = "disconnected"
    
    return {
        "message": "Sales Forecasting API - Clean Approach",
        "version": "3.0",
        "mongodb_status": mongo_status,
        "model_format": MODEL_FORMAT
    }

@app.get("/customers")
async def list_customers():
    """List available customers"""
    try:
        customers = collection.distinct("soldToPartyName")
        return {"customers": customers, "total_count": len(customers)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load customers: {str(e)}")

@app.get("/customer/{customer_name}/products")
async def list_customer_products(customer_name: str):
    """List available products for a customer"""
    try:
        products = collection.distinct("itemDescription", {"soldToPartyName": customer_name})
        return {"customer": customer_name, "products": products, "total_count": len(products)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/customer/{customer_name}/product/{product_name}/forecast")
async def customer_product_forecast(
    customer_name: str,
    product_name: str,
    horizon: int = 30,
    force_retrain: bool = False,
    skip_fridays: bool = False
):
    """
    API 1: Forecast for specific customer + product combination
    skip_fridays: if true, Saudi weekend (Friday+Saturday) will be excluded from training and from the returned predictions.
    Returns: Array of predictions with customer_name, date, product_name, predicted_count, bounds
    """
    try:
        # Train/load model for this customer + product (pass skip_fridays)
        model = train_customer_product_model(customer_name, product_name, force_retrain, skip_fridays)
        
        # Generate future predictions; if skip_fridays True, get non-weekend future rows
        if skip_fridays:
            future_forecast = _get_future_non_weekend_forecast(model, horizon)
        else:
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
            # only take the last `horizon` rows (future)
            future_forecast = forecast.tail(horizon)
        
        # Format response according to your specification and clamp negatives to 0
        response = []
        for _, row in future_forecast.iterrows():
            yhat = float(row.get('yhat', 0.0))
            yhat_upper = float(row.get('yhat_upper', 0.0))
            yhat_lower = float(row.get('yhat_lower', 0.0))

            predicted_count = round(max(0.0, yhat), 2)
            # Ensure bounds make sense and are not negative
            predicted_lower = round(max(0.0, yhat_lower), 2)
            predicted_upper = round(max(predicted_count, yhat_upper, 0.0), 2)

            response.append({
                "customer_name": customer_name,
                "date": row['ds'].strftime('%Y-%m-%d'),
                "product_name": product_name,
                "predicted_count": predicted_count,
                "predicted_upper_bound": predicted_upper,
                "predicted_lower_bound": predicted_lower
            })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/customer/{customer_name}/forecast")
async def customer_total_forecast(
    customer_name: str,
    horizon: int = 30,
    force_retrain: bool = False,
    skip_fridays: bool = False
):
    """
    API 2: Forecast for customer total (all products aggregated)
    skip_fridays: if true, Saudi weekend (Friday+Saturday) will be excluded from training and from the returned predictions.
    Returns: Array of predictions with customer_name, date, predicted_count, bounds (no product_name)
    """
    try:
        # Train/load model for this customer's aggregated data (pass skip_fridays)
        model = train_customer_total_model(customer_name, force_retrain, skip_fridays)
        
        # Generate future dates
        if skip_fridays:
            future_forecast = _get_future_non_weekend_forecast(model, horizon)
        else:
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
            future_forecast = forecast.tail(horizon)
        
        # Format response (clamp negatives to 0)
        response = []
        for _, row in future_forecast.iterrows():
            yhat = float(row.get('yhat', 0.0))
            yhat_upper = float(row.get('yhat_upper', 0.0))
            yhat_lower = float(row.get('yhat_lower', 0.0))

            predicted_count = round(max(0.0, yhat), 2)
            predicted_lower = round(max(0.0, yhat_lower), 2)
            predicted_upper = round(max(predicted_count, yhat_upper, 0.0), 2)

            response.append({
                "customer_name": customer_name,
                "date": row['ds'].strftime('%Y-%m-%d'),
                "predicted_count": predicted_count,
                "predicted_upper_bound": predicted_upper,
                "predicted_lower_bound": predicted_lower
            })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/cache/info")
async def cache_info():
    """Get information about cached models"""
    try:
        if not os.path.exists(MODEL_CACHE_DIR):
            return {"cached_models": [], "total_count": 0}
        
        cached_files = []
        for filename in os.listdir(MODEL_CACHE_DIR):
            filepath = os.path.join(MODEL_CACHE_DIR, filename)
            if os.path.isfile(filepath) and filename.endswith('.joblib'):
                try:
                    stat = os.stat(filepath)
                    file_info = {
                        "filename": filename,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                    }
                    
                    # Determine model type from filename
                    if "_product_" in filename:
                        file_info["model_type"] = "customer_product"
                    else:
                        file_info["model_type"] = "customer_total"
                    
                    cached_files.append(file_info)
                except Exception as e:
                    cached_files.append({"filename": filename, "error": str(e)})
        
        return {
            "cached_models": cached_files,
            "total_count": len(cached_files),
            "cache_directory": MODEL_CACHE_DIR
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read cache info: {str(e)}")

@app.delete("/models/cache/clear")
async def clear_cache():
    """Clear all cached models"""
    try:
        if not os.path.exists(MODEL_CACHE_DIR):
            return {"message": "Cache directory doesn't exist"}
        
        deleted_files = []
        for filename in os.listdir(MODEL_CACHE_DIR):
            filepath = os.path.join(MODEL_CACHE_DIR, filename)
            if os.path.isfile(filepath) and filename.endswith('.joblib'):
                os.remove(filepath)
                deleted_files.append(filename)
        
        return {
            "message": f"Cleared {len(deleted_files)} cached models",
            "deleted_files": deleted_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/database/stats")
async def database_stats():
    """Get database statistics"""
    try:
        total_documents = collection.count_documents({})
        unique_customers = len(collection.distinct("soldToPartyName"))
        unique_products = len(collection.distinct("itemDescription"))
        
        return {
            "total_documents": total_documents,
            "unique_customers": unique_customers,
            "unique_products": unique_products,
            "database": DATABASE_NAME,
            "collection": COLLECTION_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

if __name__ == "__main__":
    print("Starting Sales Forecasting API - Clean Approach")
    print(f"Database: {DATABASE_NAME}.{COLLECTION_NAME}")
    print(f"Model format: {MODEL_FORMAT}")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
