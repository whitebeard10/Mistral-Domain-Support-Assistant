import json
import random
import os

def generate_dummy_data(output_path: str, num_samples: int = 500):
    """Generates a larger, more diverse dataset for better evaluation."""
    
    categories = {
        "technical": [
            ("How do I fix error code {code}?", "Error code {code} usually indicates a {reason}. Please try {fix} or restart the service."),
            ("The API is returning a {status} error.", "A {status} error happens when {explanation}. Check your {check} and try again."),
            ("My {component} is not responding.", "If the {component} is frozen, try clearing the cache in {folder} or re-installing the driver."),
            ("How to install the {sdk} SDK?", "You can install the {sdk} SDK via {manager} using the command `{command}`."),
            ("Where can I find my API keys?", "Your API keys are located in the 'Developer' tab of your dashboard under 'Security Settings'.")
        ],
        "billing": [
            ("I was overcharged for my {plan} subscription.", "Sorry for the trouble! Please email billing@example.com with your invoice number. We will process a refund for the {plan} difference."),
            ("How do I cancel my {plan} plan?", "To cancel your {plan} plan, go to Settings > Billing and click 'Cancel Subscription' at the bottom of the page."),
            ("Can I get a refund for the last {month}?", "Refunds for {month} are available if requested within 30 days. Please submit a request through the portal."),
            ("Do you accept {method}?", "Yes, we accept {method} along with all major credit cards and PayPal.")
        ],
        "account": [
            ("I forgot my password for account {user}.", "No problem! Go to the login page and click 'Forgot Password' for {user}. We'll send a reset link to your email."),
            ("How to enable 2FA?", "2FA can be enabled in your Profile Settings under 'Security'. We recommend using an app like Authy or Google Authenticator."),
            ("Can I change my username to {new_name}?", "Usernames can be changed once every 6 months. To change yours to {new_name}, visit your Profile settings."),
            ("My account {user} was hacked.", "Please immediately use the 'Secure My Account' link in your email or contact emergency-support@example.com.")
        ]
    }

    # Fillers for templates
    placeholders = {
        "code": ["E101", "E404", "E500", "E202", "X-99"],
        "reason": ["connection timeout", "database mismatch", "permission error", "config corruption"],
        "fix": ["flushing the DNS", "checking your firewall", "updating the library", "verifying your credentials"],
        "status": ["401 Unauthorized", "503 Service Unavailable", "429 Too Many Requests", "403 Forbidden"],
        "explanation": ["your token has expired", "the server is under heavy load", "you have exceeded your rate limit", "the resource is restricted"],
        "check": ["auth headers", "server status page", "request payload", "API quota"],
        "component": ["dashboard", "mobile app", "cli tool", "background worker"],
        "folder": ["/var/log", "/tmp/cache", "AppData/Local", "usr/local/bin"],
        "sdk": ["Python", "Node.js", "Ruby", "Go", "Java"],
        "manager": ["pip", "npm", "gem", "go get", "maven"],
        "command": ["pip install sdk", "npm install sdk", "bundle add sdk", "go get sdk/v2"],
        "plan": ["Pro", "Enterprise", "Starter", "Legacy"],
        "month": ["April", "last month", "current cycle"],
        "method": ["Apple Pay", "Google Pay", "Bitcoin", "Wire Transfer"],
        "user": ["user_123", "admin_test", "dev_account"],
        "new_name": ["SuperDev", "CloudNinja", "LogicMaster"]
    }

    instruction_templates = [
        "{instr}",
        "I have a question: {instr}",
        "Help! {instr}",
        "Could you tell me {instr}?",
        "Please help me with this: {instr}",
        "Hey, {instr}",
        "{instr} Any ideas?"
    ]

    data = []
    
    # Ensure we cover all templates at least once
    all_templates = []
    for cat in categories.values():
        all_templates.extend(cat)

    for _ in range(num_samples):
        # Pick a random template from a random category
        cat_name = random.choice(list(categories.keys()))
        instr_tpl, resp_tpl = random.choice(categories[cat_name])
        
        # Fill placeholders
        local_data = {k: random.choice(v) for k, v in placeholders.items()}
        
        instr = instr_tpl.format(**local_data)
        resp = resp_tpl.format(**local_data)
        
        # Apply instruction variety
        if random.random() > 0.3: # 70% chance to use a wrapper template
            final_instr = random.choice(instruction_templates).format(instr=instr.lower().strip('?'))
            if not final_instr.endswith('?'):
                final_instr += '?'
        else:
            final_instr = instr

        data.append({
            "instruction": final_instr,
            "response": resp
        })

    # Shuffle to mix categories
    random.shuffle(data)
        
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    count = 500
    generate_dummy_data("data/raw/raw_support_data.jsonl", count)
    print(f"Generated {count} diverse dummy samples in data/raw/raw_support_data.jsonl")
