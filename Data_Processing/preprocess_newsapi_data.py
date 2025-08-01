import csv
from datetime import datetime

# Input and output filenames
input_filename = f"newsapi.csv"
output_filename = "cleaned_microsoft_stock_news.csv"

# Set to store unique content
unique_contents = set()

# Read and filter duplicates
with open(input_filename, "r", encoding="utf-8") as infile, open(output_filename, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Read header and write to new file
    header = next(reader)
    writer.writerow(header)
    
    for row in reader:
        date, content = row
        date = date.split("T")[0]  # Remove time, keep only date
        if content not in unique_contents:
            unique_contents.add(content)
            writer.writerow([date, content])

print(f"Cleaned data saved to {output_filename}")
