from time import sleep

from tqdm import tqdm
while True:
    fruits = tqdm(enumerate(["apple", "orange", "grape","apple", "orange", "grape","apple", "orange", "grape","apple", "orange", "grape","apple", "orange", "grape","apple", "orange", "grape","apple", "orange", "grape","apple", "orange", "grape","apple", "orange", "grape","apple", "orange", "grape",]))
    for fruit in fruits:
        sleep(0.5)
        fruits.set_description(f"Picking {fruit},Picking {fruit},Picking {fruit},Picking {fruit}")