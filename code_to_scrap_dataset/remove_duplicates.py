import json

file_path = 'metu_dataset.json'
# Load the data
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Original number of entries: {len(data)}")

# Use dictionary to keep only unique content, keeping first URL for each content
content_dict = {}
duplicates_found = 0

for entry in data:
    content = entry['content']
    if content not in content_dict:
        content_dict[content] = entry
    else:
        duplicates_found += 1
        print(f"Found exact duplicate:")
        print(f"Original URL: {content_dict[content]['URL']}")
        print(f"Duplicate URL: {entry['URL']}\n")

# Convert back to list
cleaned_data = list(content_dict.values())

print(f"Total exact duplicates removed: {duplicates_found}")
print(f"Number of entries after removing duplicates: {len(cleaned_data)}")

# Save cleaned data
with open('metu_unique_content.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
