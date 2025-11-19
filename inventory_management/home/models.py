from django.db import models

class Store(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name


class Product(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    sku = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.sku})"


class Inventory(models.Model):
    id = models.AutoField(primary_key=True)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    quantity = models.IntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('product', 'store')

    def __str__(self):
        return f"{self.product.name} @ {self.store.name}: {self.quantity}"


class Sale(models.Model):
    id = models.AutoField(primary_key=True)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    date = models.DateField()
    quantity_sold = models.IntegerField()

    class Meta:
        unique_together = ('product', 'store', 'date')

    def save(self, *args, **kwargs):
        # Only adjust inventory when creating a new sale (not updating)
        if not self.pk:
            inventory, created = Inventory.objects.get_or_create(
                product=self.product,
                store=self.store,
                defaults={"quantity": 0}
            )

            if inventory.quantity < self.quantity_sold:
                raise ValueError(
                    f"Not enough inventory for product {self.product.name} at {self.store.name}"
                )

            inventory.quantity -= self.quantity_sold
            inventory.save()

        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.product.name} - {self.date} - Sold: {self.quantity_sold}"



class Purchase(models.Model):
    id = models.AutoField(primary_key=True)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    date = models.DateField()
    quantity_added = models.IntegerField()

    def __str__(self):
        return f"Restock {self.product.name} +{self.quantity_added}"

class DemandRequest(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    store = models.ForeignKey(Store, on_delete=models.CASCADE)
    date = models.DateField()

    def __str__(self):
        return f"Demand Request: {self.product.name} @ {self.store.name} on {self.date}"