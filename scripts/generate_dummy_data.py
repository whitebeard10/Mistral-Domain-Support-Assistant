import json
import random

def generate_dummy_data(output_path: str, num_samples: int = 100):
    scenarios = [
        {
            "instruction": "How do I reset my password?",
            "response": "To reset your password, click on the 'Forgot Password' link on the login page. You will receive an email with instructions to create a new password. Make sure to check your spam folder if you don't see the email within a few minutes."
        },
        {
            "instruction": "My account is locked. What should I do?",
            "response": "If your account is locked due to multiple failed login attempts, it will automatically unlock after 30 minutes. Alternatively, you can contact our support team at support@example.com to request an immediate unlock after verifying your identity."
        },
        {
            "instruction": "How can I update my billing information?",
            "response": "You can update your billing information by navigating to the 'Settings' section of your dashboard and selecting 'Billing'. From there, you can add or remove credit cards and update your billing address."
        },
        {
            "instruction": "What is your refund policy?",
            "response": "We offer a full refund within 30 days of purchase for any reason. To request a refund, please submit a ticket through our support portal with your order number and the reason for the request."
        },
        {
            "instruction": "How do I integrate the API into my Python project?",
            "response": "To integrate our API, first install the client library using `pip install example-sdk`. Then, initialize the client with your API key: `from example import Client; client = Client(api_key='your_key')`. You can then use the client to make requests."
        }
    ]
    
    data = []
    for _ in range(num_samples):
        item = random.choice(scenarios)
        # Add some variation
        data.append(item)
        
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    generate_dummy_data("data/raw/raw_support_data.jsonl", 200)
    print("Generated dummy raw data.")
