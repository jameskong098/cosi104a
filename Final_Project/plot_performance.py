import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

# Data
rounds = [1, 2, 3, 4]
scores = [0, 0, 0, 0]

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(rounds, scores, marker='o')

# Add titles and labels
plt.title('Performance Chart')
plt.xlabel('Rounds')
plt.ylabel('Score')

# Save the plot
plt.grid(True)
plt.savefig('performance_chart.png')