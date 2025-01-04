'''
mc2pt.py
Converting .mc files to .pt files.
You can make your own converter, i.e., .osz converter.
Usage: python mc2pt.py
'''
import json
import torch
import os
import re
import csv
from collections import Counter

def extract_keys_lv(s):
    pattern = r'(\d+)K\s+(\S+)\s+Lv\.(\d+)'
    match = re.match(pattern, s)
    if match:
        keys = int(match.group(1))
        lv = int(match.group(3))
        return keys, lv
    else:
        return None, None

def clean_creator(s):
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s)
    return cleaned.lower()

def calculate_time_step(beat, base_resolution):
    measure, numerator, denominator = beat
    return int(measure * base_resolution + ((numerator + 1) / denominator) * base_resolution)

def reshape_notes(tensor, target_row=1000):
    current_n = tensor.shape[2]
    if current_n > target_row:
        return tensor[:, :, :target_row]
    else:
        padding = torch.zeros((tensor.shape[0], tensor.shape[1], target_row - current_n), dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=2)

def process_4key_chart(json_file, output_folder, base_resolution=192):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    mode = data.get('meta', {}).get('mode', 'Unknown')
    if mode != 0:
        return 0
    version = data.get('meta', {}).get('version', 'Unknown')
    keys, lv = extract_keys_lv(version)
    if keys is None or keys != 4 or lv < 15:
        return 0
    title = data.get('meta', {}).get('song', {}).get('title', 'Unknown')
    artist = data.get('meta', {}).get('song', {}).get('artist', 'Unknown')
    creator = data.get('meta', {}).get('creator', 'Unknown')
    cleaned_creator = clean_creator(creator)
    cid = data.get('meta', {}).get('id', 'Unknown')
    notes = data.get('note', [])
    processed_notes = []
    for note in notes:
        if 'beat' in note:
            time_step = calculate_time_step(note['beat'], base_resolution)
            processed_notes.append((time_step, note, False))
        if 'endbeat' in note:
            end_time_step = calculate_time_step(note['endbeat'], base_resolution)
            processed_notes.append((end_time_step, note, True))
    processed_notes.sort(key=lambda x: x[0])
    max_time_step = max(x[0] for x in processed_notes) + 1
    notes_tensor = torch.zeros((3, 4, max_time_step), dtype=torch.int8)
    last_time_step = -1
    current_row = 0
    for time_step, note, is_end in processed_notes:
        if 'column' not in note:
            continue
        col = note['column']
        if col < 0 or col > 3:
            return 0
        if time_step == last_time_step:
            row = current_row
        else:
            current_row += 1
            row = current_row
        if is_end:
            notes_tensor[2, col, row] = 1
        else:
            if 'endbeat' in note:
                notes_tensor[1, col, row] = 1
            else:
                notes_tensor[0, col, row] = 1
        last_time_step = time_step
    non_empty_indices = torch.where(torch.any(notes_tensor, dim=(0, 1)))[0]
    notes_tensor = reshape_notes(notes_tensor[:, :, non_empty_indices])
    output_path = os.path.join(output_folder, f'{cid}.pt')
    torch.save({'notes': notes_tensor, 'creator': cleaned_creator, 'cid': cid}, output_path)
    print(f'Processed {creator}\'s chart \'{title}\', \'{version}\' successfully.')
    return creator

def process_all_mc_files(input_folder, output_folder):
    sum = 0
    creator_count = Counter()
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.mc'):
                input_path = os.path.join(root, file)
                creator = process_4key_chart(input_path, output_folder)
                if creator:
                    creator_count[creator] += 1
                    sum += 1

    print(f'Successfully processed {sum} files.')
    sorted_creators = sorted(creator_count.items(), key=lambda x: x[1], reverse=True)
    with open('creators.csv', 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Creator', 'Count'])
        writer.writerows(sorted_creators)

if __name__ == '__main__':
    input_folder = 'beatmap'
    output_folder = 'processed_tensors'
    os.makedirs(output_folder, exist_ok=True)
    process_all_mc_files(input_folder, output_folder)
