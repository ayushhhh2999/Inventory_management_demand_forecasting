from django.contrib import admin
from .models import Store, Product, Inventory, Sale, Purchase


# ---------------------------
# STORE ADMIN
# ---------------------------
@admin.register(Store)
class StoreAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "location")
    search_fields = ("name", "location")
    ordering = ("id",)
    list_per_page = 25


# ---------------------------
# PRODUCT ADMIN
# ---------------------------
@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "sku", "category")
    search_fields = ("name", "sku", "category")
    ordering = ("id",)
    list_filter = ("category",)
    list_per_page = 25


# ---------------------------
# INVENTORY ADMIN
# ---------------------------
@admin.register(Inventory)
class InventoryAdmin(admin.ModelAdmin):
    list_display = ("id", "product", "store", "quantity", "updated_at")
    search_fields = ("product__name", "store__name")
    list_filter = ("store", "product")
    ordering = ("-updated_at",)
    readonly_fields = ("updated_at",)

    list_editable = ("quantity",)   # <-- THIS ENABLES DIRECT EDIT
    list_per_page = 25



# ---------------------------
# SALES ADMIN
# ---------------------------
@admin.register(Sale)
class SaleAdmin(admin.ModelAdmin):
    list_display = ("id", "product", "store", "date", "quantity_sold")
    list_filter = ("store", "product", "date")
    search_fields = ("product__name", "store__name")
    ordering = ("-date",)
    list_per_page = 25


# ---------------------------
# PURCHASE ADMIN
# ---------------------------
@admin.register(Purchase)
class PurchaseAdmin(admin.ModelAdmin):
    list_display = ("id", "product", "store", "date", "quantity_added")
    list_filter = ("store", "product", "date")
    search_fields = ("product__name", "store__name")
    ordering = ("-date",)
    list_per_page = 25
