from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    StoreViewSet, ProductViewSet, InventoryViewSet,
    SaleViewSet, PurchaseViewSet,
    remove_store, remove_product, bulk_create_products,add_sale,
    bulk_create_stores,predict_demand,get_inventory,get_sales
)

router = DefaultRouter()
router.register(r'stores', StoreViewSet)
router.register(r'products', ProductViewSet)
router.register(r'inventory', InventoryViewSet)
router.register(r'sales', SaleViewSet)
router.register(r'purchases', PurchaseViewSet)

urlpatterns = [
    # ✅ custom routes FIRST
    path('api/products/bulk-create/', bulk_create_products),
    path('api/remove-store/', remove_store),
    path('api/remove-product/', remove_product),
    path('api/sales/add/', add_sale),
    path('api/stores/bulk-create/', bulk_create_stores),
    path('api/predict-demand/', predict_demand),
    path("api/inventory/view/", get_inventory),
    path("api/sales/view/", get_sales),

    # ✅ router routes LAST
    path('api/', include(router.urls)),
]
