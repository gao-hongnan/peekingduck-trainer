import string
import random


class Order:
    def __init__(self):
        self.id = "".join(random.choices(string.ascii_lowercase, k=6))
        self.status = "open"

    def set_status(self, status):
        self.status = status


class Authorizer_SMS:
    def __init__(self):
        self.authorized = False
        self.code = None

    def generate_sms_code(self):
        self.code = "".join(random.choices(string.digits, k=6))
        print(f"Generated SMS code: {self.code}")

    def authorize(self):
        code = input("Enter SMS code: ")
        self.authorized = code == self.code

    def is_authorized(self) -> bool:
        return self.authorized


class PaymentProcessor:
    def pay(self, order):
        authorizer = Authorizer_SMS()
        authorizer.generate_sms_code()
        authorizer.authorize()
        if not authorizer.is_authorized():
            raise Exception("Not authorized")
        print(f"Processing payment for order with id {order.id}")
        order.set_status("paid")


if __name__ == "__main__":
    order = Order()
    processor = PaymentProcessor()
    processor.pay(order)
    print(f"Order {order.id} is {order.status}")
