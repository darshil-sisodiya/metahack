import requests

URL = "http://127.0.0.1:8000"

# Reset first
requests.post(f"{URL}/reset")

for i in range(10):
    res = requests.post(
        f"{URL}/step",
        json={
            "steam_valve": 50,
            "reflux_ratio": 50,
            "feed_rate": 50,
            "vent": 0
        }
    )

    data = res.json()
    obs = data["observation"]

    print(f"Step {i+1}:")
    print(f"  Temp: {obs['temperature']:.2f}")
    print(f"  Pressure: {obs['pressure']:.2f}")
    print(f"  Reward: {data['reward']}")
    print()