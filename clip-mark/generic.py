import os
import json

def calculate_average_difference(directory):
    total_difference = 0
    file_count = 0

    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    original_count = data.get('original_count', 0)
                    adv_count = data.get('adversarial_count', 0)
                    total_difference += (original_count - adv_count)
                    file_count += 1

    if file_count == 0:
        return 0

    average_difference = total_difference / file_count
    return average_difference

directory = 'adv_images'
average_difference = calculate_average_difference(directory)
print(f'Average difference: {average_difference}')