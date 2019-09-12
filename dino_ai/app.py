from scipy import stats
import matplotlib.pyplot as plt

data = stats.bernoulli.rvs(0.7,size=100)

print(data)

plt.hist(data)

plt.show()
