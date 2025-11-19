from rest_framework.decorators import api_view
from rest_framework import viewsets, status
from rest_framework.response import Response

from .models import Product, Store, Inventory, Sale, Purchase, DemandRequest
from .serializers import (
    ProductSerializer, StoreSerializer, InventorySerializer,
    SaleSerializer, PurchaseSerializer
)
import numpy as np
from datetime import datetime
import holidays
import joblib
import xgboost as xgb
from datetime import datetime
import pandas as pd
# -------------------------
# CRUD ViewSets
# -------------------------

class StoreViewSet(viewsets.ModelViewSet):
    queryset = Store.objects.all()
    serializer_class = StoreSerializer


class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer


class InventoryViewSet(viewsets.ModelViewSet):
    queryset = Inventory.objects.all().select_related("product", "store")
    serializer_class = InventorySerializer


class SaleViewSet(viewsets.ModelViewSet):
    queryset = Sale.objects.all().order_by("-date")
    serializer_class = SaleSerializer


class PurchaseViewSet(viewsets.ModelViewSet):
    queryset = Purchase.objects.all().order_by("-date")
    serializer_class = PurchaseSerializer


# -------------------------
# Custom APIs
# -------------------------
import numpy as np
from datetime import datetime
import holidays
import joblib
import xgboost as xgb
import os
from django.conf import settings

# -----------------------------
# LOAD MODEL + SCALER ONCE
# -----------------------------

# Paths
MODEL_PATH = os.path.join(settings.BASE_DIR, "xgboost_sales_model.json")
SCALER_PATH = os.path.join(settings.BASE_DIR, "scaler.pkl")

# Load XGBoost Booster model from JSON
xgboost_model = xgb.Booster()
xgboost_model.load_model(MODEL_PATH)

# Load StandardScaler
scaler = joblib.load(SCALER_PATH)

india_holidays = holidays.country_holidays('IN')


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def weekend_or_weekday_func(year, month, day):
    d = datetime(year, month, day)
    return 1 if d.weekday() > 4 else 0

def which_day_func(year, month, day):
    return datetime(year, month, day).weekday()


def predict_stock_need(date_str, item_num, store_num):
    # Parse date
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    # Feature engineering
    weekend = weekend_or_weekday_func(year, month, day)
    is_holiday = 1 if india_holidays.get(date_str) else 0

    m1 = np.sin(month * (2 * np.pi / 12))
    m2 = np.cos(month * (2 * np.pi / 12))

    weekday = which_day_func(year, month, day)

    # Input data
    input_data = np.array([
        [store_num, item_num, month, day, weekend, is_holiday, m1, m2, weekday]
    ])

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Convert to DMatrix
    dmatrix_input = xgb.DMatrix(scaled_input)

    # Predict
    prediction = xgboost_model.predict(dmatrix_input)

    return max(0, round(prediction[0]))


@api_view(['POST'])
def remove_store(request):
    store_id = request.data.get("id")

    if not store_id:
        return Response({"error": "Store ID is required"}, status=400)

    try:
        store = Store.objects.get(id=store_id)
        store.delete()
        return Response({"message": "Store deleted successfully"}, status=200)

    except Store.DoesNotExist:
        return Response({"error": "Store not found"}, status=404)


@api_view(['POST'])
def remove_product(request):
    product_id = request.data.get("id")

    if not product_id:
        return Response({"error": "Product ID is required"}, status=400)

    try:
        product = Product.objects.get(id=product_id)
        product.delete()
        return Response({"message": "Product deleted successfully"}, status=200)

    except Product.DoesNotExist:
        return Response({"error": "Product not found"}, status=404)


@api_view(['POST'])
def bulk_create_products(request):
    products = request.data.get("products", [])

    if not products:
        return Response({"error": "No products provided"}, status=400)

    serializer = ProductSerializer(data=products, many=True)
    if serializer.is_valid():
        serializer.save()
        return Response({
            "message": "Products created successfully",
            "count": len(products),
            "products": serializer.data
        }, status=201)

    return Response(serializer.errors, status=400)


@api_view(['POST'])
def add_sale(request):
    sku = request.data.get("sku")
    store_name = request.data.get("store_name")  # <-- store_name instead of store_id
    date = request.data.get("date")
    quantity = int(request.data.get("quantity", 1))

    if not (sku and store_name and date):
        return Response(
            {"error": "sku, store_name, and date are required"},
            status=400
        )

    # Get Product (using SKU)
    try:
        product = Product.objects.get(sku=sku)
    except Product.DoesNotExist:
        return Response({"error": f"Product with SKU '{sku}' not found"}, 404)

    # Get Store (using store_name)
    try:
        store = Store.objects.get(name=store_name)
    except Store.DoesNotExist:
        return Response({"error": f"Store '{store_name}' not found"}, 404)

    # Inventory must be correct product+store
    inventory, created = Inventory.objects.get_or_create(
        product=product,
        store=store,
        defaults={"quantity": 0}
    )

    if inventory.quantity < quantity:
        return Response({
            "error": "Not enough inventory",
            "available": inventory.quantity
        }, 400)

    # Reduce inventory
    inventory.quantity -= quantity
    inventory.save()

    # Create sale record
    data = {
        "product": product.id,
        "store": store.id,
        "date": date,
        "quantity_sold": quantity
    }
    retrain_model_with_new_sale(
        store_id=store.id,
        product_id=product.id,
        date=date,
        quantity=quantity
    )
    serializer = SaleSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
        return Response({
            "message": "Sale added successfully",
            "sale": serializer.data,
            "updated_inventory": inventory.quantity
        }, 201)

    return Response(serializer.errors, 400)




@api_view(['POST'])
def bulk_create_stores(request):
    stores = request.data.get("stores", [])

    if not stores:
        return Response({"error": "No stores provided"}, status=400)

    serializer = StoreSerializer(data=stores, many=True)
    if serializer.is_valid():
        serializer.save()
        return Response({
            "message": "Stores created successfully",
            "count": len(stores),
            "stores": serializer.data
        }, status=201)

    return Response(serializer.errors, status=400)


@api_view(['POST'])
def predict_demand(request):
    """
    Predict demand based on product name and store name.
    """

    product_name = request.data.get("product_name")
    store_name = request.data.get("store_name")
    date_str = request.data.get("date")

    # Validate request body
    if not (product_name and store_name and date_str):
        return Response(
            {"error": "product_name, store_name, and date are required"},
            status=400
        )

    # Convert names → IDs
    try:
        product = Product.objects.get(name=product_name)
    except Product.DoesNotExist:
        return Response({"error": f"Product '{product_name}' not found"}, status=404)

    try:
        store = Store.objects.get(name=store_name)
    except Store.DoesNotExist:
        return Response({"error": f"Store '{store_name}' not found"}, status=404)

    # Use IDs in model prediction
    product_id = product.id
    store_id = store.id

    # Make prediction
    predicted_value = predict_stock_need(date_str, product_id, store_id)

    return Response({
        "product": product_name,
        "store": store_name,
        "predicted_demand": predicted_value,
        "date": date_str
    })

@api_view(['POST'])
def add_inventory(request):
    """
    Add inventory using product_name + store_name + quantity.
    Auto-creates inventory entry if missing.
    """

    product_name = request.data.get("product_name")
    store_name = request.data.get("store_name")
    quantity = request.data.get("quantity")

    if not (product_name and store_name and quantity):
        return Response(
            {"error": "product_name, store_name, and quantity are required"},
            status=400
        )

    # Get product
    try:
        product = Product.objects.get(name=product_name)
    except Product.DoesNotExist:
        return Response({"error": f"Product '{product_name}' not found"}, 404)

    # Get store
    try:
        store = Store.objects.get(name=store_name)
    except Store.DoesNotExist:
        return Response({"error": f"Store '{store_name}' not found"}, 404)

    # Create or update inventory
    inventory, created = Inventory.objects.get_or_create(
        product=product,
        store=store,
        defaults={"quantity": quantity}
    )

    if not created:
        inventory.quantity += int(quantity)
        inventory.save()

    return Response({
        "message": "Inventory updated" if not created else "Inventory created",
        "inventory": {
            "product": product.name,
            "store": store.name,
            "quantity": inventory.quantity
        }
    })


@api_view(['POST'])
def bulk_add_inventory(request):
    """
    Bulk add inventory entries.
    Example input:
    {
        "items": [
            {"product_name": "Laptop", "store_name": "Delhi Store", "quantity": 20},
            {"product_name": "Mouse", "store_name": "Mumbai Store", "quantity": 15}
        ]
    }
    """
    items = request.data.get("items", [])
    if not items:
        return Response({"error": "items list is required"}, 400)

    created_count = 0
    updated_count = 0

    for item in items:
        product_name = item.get("product_name")
        store_name = item.get("store_name")
        quantity = item.get("quantity", 0)

        # Skip incomplete rows
        if not (product_name and store_name and quantity):
            continue

        try:
            product = Product.objects.get(name=product_name)
            store = Store.objects.get(name=store_name)
        except:
            continue

        inventory, created = Inventory.objects.get_or_create(
            product=product,
            store=store,
            defaults={"quantity": quantity}
        )

        if created:
            created_count += 1
        else:
            inventory.quantity += int(quantity)
            inventory.save()
            updated_count += 1

    return Response({
        "message": "Bulk inventory update completed",
        "created": created_count,
        "updated": updated_count
    }, status=201)


@api_view(['GET'])
def get_inventory(request):
    product_name = request.query_params.get("product_name")
    store_name = request.query_params.get("store_name")

    if not (product_name and store_name):
        return Response(
            {"error": "product_name and store_name are required"},
            status=400
        )

    # Fetch Product
    try:
        product = Product.objects.get(name=product_name)
    except Product.DoesNotExist:
        return Response({"error": f"Product '{product_name}' not found"}, 404)

    # Fetch Store
    try:
        store = Store.objects.get(name=store_name)
    except Store.DoesNotExist:
        return Response({"error": f"Store '{store_name}' not found"}, 404)

    # Fetch Inventory
    inventory, created = Inventory.objects.get_or_create(
        product=product, store=store
    )

    return Response({
        "product": product.name,
        "store": store.name,
        "quantity": inventory.quantity,
        "updated_at": inventory.updated_at
    })

@api_view(['POST'])
def get_inventory(request):
    product_name = request.data.get("product_name")
    store_name = request.data.get("store_name")

    if not (product_name and store_name):
        return Response(
            {"error": "product_name and store_name are required"},
            status=400
        )

    # Fetch Product
    try:
        product = Product.objects.get(name=product_name)
    except Product.DoesNotExist:
        return Response({"error": f"Product '{product_name}' not found"}, 404)

    # Fetch Store
    try:
        store = Store.objects.get(name=store_name)
    except Store.DoesNotExist:
        return Response({"error": f"Store '{store_name}' not found"}, 404)

    # Fetch Inventory
    inventory, created = Inventory.objects.get_or_create(
        product=product, store=store
    )

    return Response({
        "product": product.name,
        "store": store.name,
        "quantity": inventory.quantity,
        "updated_at": inventory.updated_at
    })

@api_view(['POST'])
def get_sales(request):
    product_name = request.data.get("product_name")
    store_name = request.data.get("store_name")

    if not (product_name and store_name):
        return Response(
            {"error": "product_name and store_name are required"},
            status=400
        )

    # Fetch Product
    try:
        product = Product.objects.get(name=product_name)
    except Product.DoesNotExist:
        return Response({"error": f"Product '{product_name}' not found"}, 404)

    # Fetch Store
    try:
        store = Store.objects.get(name=store_name)
    except Store.DoesNotExist:
        return Response({"error": f"Store '{store_name}' not found"}, 404)

    # Fetch all sales that match
    sales = Sale.objects.filter(product=product, store=store).order_by("-date")

    serializer = SaleSerializer(sales, many=True)

    return Response({
        "product": product.name,
        "store": store.name,
        "total_sales": len(sales),
        "sales": serializer.data
    })


def create_features_from_sale(store_id, product_id, date, quantity):
    """
    Create X, y rows exactly like original training DataFrame.
    """

    date_obj = datetime.strptime(date, "%Y-%m-%d")

    df = pd.DataFrame([{
        "store": store_id,
        "item": product_id,
        "month": date_obj.month,
        "day": date_obj.day,
        "weekend": 1 if date_obj.weekday() >= 5 else 0,
        "holidays": 0,   # <-- FIX: must match original training exactly
        "m1": np.sin(date_obj.month * (2 * np.pi / 12)),
        "m2": np.cos(date_obj.month * (2 * np.pi / 12)),
        "weekday": date_obj.weekday(),
        "target": quantity
    }])

    X = df[["store", "item", "month", "day", "weekend", "holidays", "m1", "m2", "weekday"]]
    y = df["target"]

    return X, y


def retrain_model_with_new_sale(store_id, product_id, date, quantity):
    # Load existing model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Create new feature row
    X_new, y_new = create_features_from_sale(store_id, product_id, date, quantity)

    # Scale X_new
    X_scaled = scaler.transform(X_new)

    # Incremental training
    model.fit(
        X_scaled,
        y_new,
        xgb_model=MODEL_PATH
    )

    # Save updated model
    model.save_model(MODEL_PATH)
    print("Model retrained with new sale data.")