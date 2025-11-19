from rest_framework import serializers
from .models import Product, Store, Inventory, Sale, Purchase


class StoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Store
        fields = '__all__'


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'


class InventorySerializer(serializers.ModelSerializer):
    # Read-only nested objects
    product = ProductSerializer(read_only=True)
    store = StoreSerializer(read_only=True)

    # Write-only IDs for create/update
    product_id = serializers.IntegerField(write_only=True)
    store_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = Inventory
        fields = [
            'id',
            'product',
            'store',
            'quantity',
            'updated_at',
            'product_id',
            'store_id'
        ]

    # FIX 1: How to create Inventory using product_id, store_id
    def create(self, validated_data):
        product_id = validated_data.pop("product_id")
        store_id = validated_data.pop("store_id")

        return Inventory.objects.create(
            product_id=product_id,
            store_id=store_id,
            **validated_data
        )

    # FIX 2: How to update Inventory using product_id, store_id
    def update(self, instance, validated_data):
        instance.product_id = validated_data.get("product_id", instance.product_id)
        instance.store_id = validated_data.get("store_id", instance.store_id)
        instance.quantity = validated_data.get("quantity", instance.quantity)

        instance.save()
        return instance


class SaleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sale
        fields = '__all__'


class PurchaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Purchase
        fields = '__all__'
