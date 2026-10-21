from rest_framework.test import APITestCase
from rest_framework import status
from django.urls import reverse

class InventoryAPITest(APITestCase):

    def setUp(self):
        self.base_url = "/api"

    def test_full_flow(self):
        # 1. Create Store
        store_res = self.client.post(f"{self.base_url}/stores/", {
            "name": "Delhi Store",
            "location": "Delhi"
        }, format="json")

        self.assertEqual(store_res.status_code, status.HTTP_201_CREATED)
        store_id = store_res.data["id"]

        # 2. Create Product
        product_res = self.client.post(f"{self.base_url}/products/", {
            "name": "Laptop",
            "sku": "LAP12345",
            "category": "Electronics"
        }, format="json")

        self.assertEqual(product_res.status_code, status.HTTP_201_CREATED)
        product_id = product_res.data["id"]

        # 3. Add Inventory
        inventory_res = self.client.post(f"{self.base_url}/inventory/", {
            "product_id": product_id,
            "store_id": store_id,
            "quantity": 50
        }, format="json")

        self.assertEqual(inventory_res.status_code, status.HTTP_201_CREATED)

        # 4. Add Sale
        sale_res = self.client.post(f"{self.base_url}/sales/add/", {
            "sku": "LAP12345",
            "store_name": "Delhi Store",
            "date": "2026-05-05",
            "quantity": 2
        }, format="json")

        self.assertEqual(sale_res.status_code, status.HTTP_201_CREATED)

        # 5. Get Inventory
        inventory_check = self.client.post(f"{self.base_url}/inventory/view/", {
            "product_name": "Laptop",
            "store_name": "Delhi Store"
        }, format="json")

        self.assertEqual(inventory_check.status_code, status.HTTP_200_OK)
        self.assertTrue("quantity" in inventory_check.data)

        # 6. Predict Demand
        predict_res = self.client.post(f"{self.base_url}/predict-demand/", {
            "product_name": "Laptop",
            "store_name": "Delhi Store",
            "date": "2026-05-10"
        }, format="json")

        self.assertEqual(predict_res.status_code, status.HTTP_200_OK)
        self.assertTrue("predicted_demand" in predict_res.data)