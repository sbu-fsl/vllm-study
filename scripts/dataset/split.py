import csv
import sys

# Increase field size limit (important for your dataset)
csv.field_size_limit(sys.maxsize)

def split_csv(input_file, column_name):
    with open(input_file, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if column_name not in fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found. "
                f"Available columns: {fieldnames}"
            )

        # Create output files
        qmsum_file = open("/mnt/gpfs/llm-datasets/qmsum.csv", "w", newline='', encoding='utf-8')
        narrativeqa_file = open("/mnt/gpfs/llm-datasets/narrativeqa.csv", "w", newline='', encoding='utf-8')

        qmsum_writer = csv.DictWriter(qmsum_file, fieldnames=fieldnames)
        narrativeqa_writer = csv.DictWriter(narrativeqa_file, fieldnames=fieldnames)

        # Write headers
        qmsum_writer.writeheader()
        narrativeqa_writer.writeheader()

        for row in reader:
            value = row[column_name]

            if value == "qmsum":
                qmsum_writer.writerow(row)
            elif value == "narrativeqa":
                narrativeqa_writer.writerow(row)

        qmsum_file.close()
        narrativeqa_file.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split.py <csv_file> <column_name>")
        sys.exit(1)

    split_csv(sys.argv[1], sys.argv[2])

