import matplotlib.pyplot as plt

smoothing = 10
scores = []
with open("SI_nodemo/SpaceInvadersScoresNoDemo200_.txt") as f:
    lines = f.readlines()
    for line in lines:
        score = float(line.strip())
        scores.append(score)

smoothed_scores = []
for i, score in enumerate(scores):
    low = max(0, i - smoothing + 1)
    smoothed_scores.append(sum(scores[low : i]) / (i - low + 1))

scores2 = []
with open("SIScores200_.txt") as f:
    lines = f.readlines()
    for line in lines:
        score = float(line.strip())
        scores2.append(score)

smoothed_scores2 = []
for i, score in enumerate(scores2):
    low = max(0, i - smoothing + 1)
    smoothed_scores2.append(sum(scores2[low : i]) / (i - low + 1))

plt.plot(range(len(smoothed_scores)), smoothed_scores)
plt.plot(range(len(smoothed_scores2)), smoothed_scores2)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend(['Without Demo', 'With Demo'], loc='upper left')
plt.title("SpaceInvaders DQN Comparison")
plt.savefig('SIComp.jpg')