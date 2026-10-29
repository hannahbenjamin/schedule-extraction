"""
Quick check that your Anthropic API key, billing and Python setup work.

Windows (PowerShell):
    $env:ANTHROPIC_API_KEY="sk-ant-...your-NEW-key..."
    pip install anthropic
    python test_api.py
"""
import os
import sys

try:
    import anthropic
except ImportError:
    sys.exit("The anthropic library isn't installed. Run:  pip install anthropic")

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit('No API key found. In PowerShell run:  $env:ANTHROPIC_API_KEY="sk-ant-..."\n'
             "(It only lasts for that window; set it again in each new window.)")

client = anthropic.Anthropic()
try:
    msg = client.messages.create(
        model="claude-fable-5-1",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say hello in five words."}],
    )
except anthropic.AuthenticationError:
    sys.exit("Key rejected. Check you copied the whole key, and that it isn't the deleted one.")
except anthropic.PermissionDeniedError as e:
    sys.exit(f"Key works but lacks access to this model or workspace: {e.message}")
except anthropic.BadRequestError as e:
    if "credit" in str(e).lower() or "billing" in str(e).lower():
        sys.exit("Key works, but the account has no credits. Add credits in the Console under Billing.")
    sys.exit(f"Request rejected: {e.message}")
except anthropic.APIConnectionError:
    sys.exit("Couldn't reach the API. Check your internet connection or any firewall/VPN.")

print("Success! The model replied:", msg.content[0].text)