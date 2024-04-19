def monty_hall(switch: bool) -> bool:
    doors = [0, 0, 1]  # 0 is goat, 1 is car
    random.shuffle(doors)
    choice = random.choice([0, 1, 2])
    revealed = next(i for i in range(3) if i != choice and doors[i] == 0)
    if switch:
        choice = next(i for i in range(3) if i != choice and i != revealed)
    return doors[choice] == 1

# Simulate 10000 trials
trials = 10000
wins_with_switch = sum(monty_hall(True) for _ in range(trials))
wins_without_switch = sum(monty_hall(False) for _ in range(trials))

print("Winning probability with switch:", wins_with_switch / trials)
print("Winning probability without switch:", wins_without_switch / trials)
