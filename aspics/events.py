def simulateEvents(today, events):
    print(f"It's {today}")
    for ev in events:
        if ev["date"] == str(today):
            print(f"so time for {ev}")
