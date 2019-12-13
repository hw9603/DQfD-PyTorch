import matplotlib.pyplot as plt

smoothing = 2
scores = []
with open("SI_screens/loss_self.txt") as f:
    lines = f.readlines()
    for line in lines:
        score = float(line.strip())
        scores.append(score)

smoothed_scores = []
for i, score in enumerate(scores):
    low = max(0, i - smoothing + 1)
    smoothed_scores.append(sum(scores[low : i]) / (i - low + 1))

scores2 = []
with open("SI_screens/loss.txt") as f:
    lines = f.readlines()
    for line in lines:
        score = float(line.strip())
        scores2.append(score)

smoothed_scores2 = []
for i, score in enumerate(scores2):
    low = max(0, i - smoothing + 1)
    smoothed_scores2.append(sum(scores2[low : i]) / (i - low + 1))

plt.plot([100 * i for i in range(len(smoothed_scores[1:]))], smoothed_scores[1:])
plt.plot([100 * i for i in range(len(smoothed_scores2[1:]))], smoothed_scores2[1:])
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend(['Without Demo', 'With Demo'], loc='upper left')
plt.title("SpaceInvaders DQN Comparison")
plt.savefig('SI_screens/SIComp_screen.jpg')