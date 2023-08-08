import pandas as pd
import numpy as np

# The list of words that could be present in a house advert
words = ['garden', 'pool', 'renovation', 'garage', 'move-in ready',
         'kitchen', 'bathroom', 'bedroom', 'living room', 'dining room']

# Generate 100 adverts
adverts = [' '.join(np.random.choice(
    words, size=np.random.randint(5, 15))) for _ in range(100)]

# Generate 100 prices within a realistic range
prices = np.random.uniform(100000, 500000, size=100)

# Generate the number of rooms and footage
rooms = np.random.randint(1, 7, size=100)
footage = np.random.randint(500, 3000, size=100)

# Create DataFrame
df = pd.DataFrame({
    'advert': adverts,
    'price': prices,
    'rooms': rooms,
    'footage': footage
})

# Save DataFrame to a CSV file
df.to_csv('your_data.csv', index=False)
print("your_data.csv generated")
